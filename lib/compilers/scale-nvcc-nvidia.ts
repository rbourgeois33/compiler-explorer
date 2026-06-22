// Copyright (c) 2018, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import * as fs from 'node:fs/promises';
import Path from 'node:path';

import Semver from 'semver';
import _ from 'underscore';

import type {CompilationInfo, CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {PTXAsmParser} from '../parsers/asm-parser-ptx.js';
import {SassAsmParser} from '../parsers/asm-parser-sass.js';
import {asSafeVer} from '../utils.js';
import {ClangParser} from './argument-parsers.js';

export class ScaleNvccNvidiaCompiler extends BaseCompiler {
    static get key() {
        return 'scale-nvcc-nvidia';
    }

    deviceAsmParser: SassAsmParser;
    ptxParser: PTXAsmParser;

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.compiler.supportsOptOutput = true;
        this.compiler.supportsDeviceAsmView = true;
        this.deviceAsmParser = new SassAsmParser(this.compilerProps);
        this.ptxParser = new PTXAsmParser(this.compilerProps);
    }

    // TODO: (for all of CUDA)
    // * lots of whitespace from nvcc
    // * would be nice to try and filter unused `.func`s from e.g. clang output

    // TEMP: -o commented out because scale can't combine an explicit `-o`
    // with `-Xcompiler=-S`. This means scale (and now nvcc too, while this
    // is shared) falls back to basename-derived default naming instead of
    // a predictable `output.s`. See findHostAsmFile()/extractDeviceCode()
    // below, which auto-detect either naming convention.
    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions?: string[]) {
        // const opts = ['-o', this.filename(outputFilename), '-g', '-lineinfo', '--keep-device-functions'];
        const opts = ['-g', '-lineinfo', '--keep-device-functions'];
        if (!filters.execute) {
            opts.push('-c', '-keep', '-keep-dir', Path.dirname(outputFilename));
            if (!filters.binary) {
                opts.push('-Xcompiler=-S');
            }
        }
        return opts;
    }

    override getArgumentParserClass() {
        return ClangParser;
    }

    override optOutputRequested(options: string[]) {
        return (
            super.optOutputRequested(options) ||
            options.includes('--optimization-info') ||
            options.includes('-opt-info')
        );
    }


    // TEMP (scale support): matches scale's per-target device output, e.g.
    // `example-cuda-nvptx64-nvidia-cuda-sm_75.s`. Captured group is the
    // arch (sm_75). The host-side file (`example.s`) never matches this,
    // since it lacks the `-cuda-nvptx64-nvidia-cuda-` infix.
    private static readonly scaleDeviceFileRe = /-cuda-nvptx64-nvidia-cuda-([^./]+)\.s$/;

    // TEMP (scale support): with `-o` omitted, find whichever `.s` file in
    // dirPath is the host output (i.e. not one of the per-arch device
    // files). Returns null if nothing matches (e.g. plain nvcc still using
    // its own default naming, or compile actually failed).
    private async findHostAsmFile(dirPath: string): Promise<string | null> {
        try {
            const files = await fs.readdir(dirPath);

            console.log('all files:', files);

            const hostFiles = files.filter(
                f =>
                    f.endsWith('.s') &&
                    !ScaleNvccNvidiaCompiler.scaleDeviceFileRe.test(f)
            );

            console.log('Host ASM candidates:', hostFiles);

            if (hostFiles.length !== 1) {
                console.warn(
                    `Expected exactly one host .s file, found ${hostFiles.length}`
                );
                return null;
            }

            return Path.join(dirPath, hostFiles[0]);
        } catch {
            return null;
        }
    }

    override async postProcess(result, outputFilename: string, filters: ParseFiltersAndOutputOptions) {
        // TEMP (scale support): outputFilename as originally computed may not
        // exist since we no longer pass `-o`. Try to recover the real file.
        if (!filters.binary && result.dirPath) {
            try {
                await fs.stat(outputFilename);
            } catch {
                const hostAsm = await this.findHostAsmFile(result.dirPath);
                if (hostAsm) {
                    try {
                        result.asmSize = (await fs.stat(hostAsm)).size;
                    } catch {
                        // leave asmSize as-is; base behaviour reports "no output" below
                    }
                    outputFilename = hostAsm;
                }
            }
        }

        const maxSize = this.env.ceProps('max-asm-size', 64 * 1024 * 1024);
        const optPromise = result.optPath ? this.processOptOutput(result.optPath) : Promise.resolve([]);
        const postProcess = _.compact(this.compiler.postProcess);
        const asmPromise = (
            filters.binary
                ? this.objdump(outputFilename, {}, maxSize, !!filters.intel, !!filters.demangle, false, false, filters)
                : (async () => {
                      if (result.asmSize === undefined) {
                          result.asm = '<No output file>';
                          return result;
                      }
                      if (result.asmSize >= maxSize) {
                          result.asm =
                              '<No output: generated assembly was too large' +
                              ` (${result.asmSize} > ${maxSize} bytes)>`;
                          return result;
                      }
                      if (postProcess.length > 0) {
                          return await this.execPostProcess(result, postProcess, outputFilename, maxSize);
                      }
                      const contents = await fs.readFile(outputFilename, {encoding: 'utf8'});
                      result.asm = contents.toString();
                      return result;
                  })()
        ).then(asm => {
            result.asm = typeof asm === 'string' ? asm : asm.asm;
            return result;
        });
        return Promise.all([asmPromise, optPromise, []]);
    }

    // Matches the start/end of a GAS inline-assembly block emitted by the host compiler.
    private static readonly appBlockStartRe = /^#APP\b/;
    private static readonly appBlockEndRe = /^#NO_APP\b/;
    // Matches the .nv_fatbin section directive that NVCC injects to hold the fat binary blob.
    private static readonly nvFatBinSectionRe = /^\s*\.section\s+\.nv_fatbin\b/;

    /**
     * Strip `#APP`/`#NO_APP` inline-assembly blocks that contain a `.nv_fatbin`
     * section from the host-side x86 assembly.  These blocks hold the raw CUDA
     * fat binary blob (the `fatbinData` label followed by hundreds of `.quad`
     * hex lines) which is never useful to inspect in the asm view.
     *
     * Only blocks that contain `.nv_fatbin` are removed; any `#APP`/`#NO_APP`
     * blocks originating from genuine user inline-assembly are left intact.
     */
    protected removeNvccFatbinaryBlob(asm: string): string {
        const lines = asm.split('\n');
        const result: string[] = [];
        let inAppBlock = false;
        let hasFatBin = false;
        let appBuffer: string[] = [];

        for (const line of lines) {
            const trimmed = line.trim();
            if (ScaleNvccNvidiaCompiler.appBlockStartRe.test(trimmed)) {
                inAppBlock = true;
                hasFatBin = false;
                appBuffer = [line];
            } else if (ScaleNvccNvidiaCompiler.appBlockEndRe.test(trimmed)) {
                inAppBlock = false;
                if (!hasFatBin) {
                    // Not a fat-binary block — keep it
                    appBuffer.push(line);
                    result.push(...appBuffer);
                }
                appBuffer = [];
            } else if (inAppBlock) {
                if (ScaleNvccNvidiaCompiler.nvFatBinSectionRe.test(line)) {
                    hasFatBin = true;
                }
                appBuffer.push(line);
            } else {
                result.push(line);
            }
        }

        // Handle (malformed) unclosed #APP block: keep it
        if (appBuffer.length > 0) {
            result.push(...appBuffer);
        }

        return result.join('\n');
    }

    override async processAsm(result, filters: ParseFiltersAndOutputOptions, options: string[]) {
        if (filters.labels && typeof result.asm === 'string') {
            result = {...result, asm: this.removeNvccFatbinaryBlob(result.asm)};
        }
        return super.processAsm(result, filters, options);
    }

    override async extractDeviceCode(
        result: CompilationResult,
        filters: ParseFiltersAndOutputOptions,
        compilationInfo: CompilationInfo,
    ) {
        const {dirPath} = result;
        const {demangle} = filters;
        const devices = {...result.devices};

        if (dirPath) {
            const files = await fs.readdir(dirPath);
            const maxSize = this.env.ceProps('max-asm-size', 64 * 1024 * 1024);

            console.log('[extractDeviceCode]: dirPath', dirPath);
            console.log('[extractDeviceCode]: all files at start', files);

            await Promise.all(
                files
                    .filter(f => ScaleNvccNvidiaCompiler.scaleDeviceFileRe.test(f))
                    .map(async name => {
                        console.log('[extractDeviceCode]: ---- processing file:', name);

                        const scaleMatch = name.match(ScaleNvccNvidiaCompiler.scaleDeviceFileRe);
                        console.log('[extractDeviceCode]: scaleMatch', scaleMatch);

                        const asm = await fs.readFile(Path.join(dirPath, name), 'utf8');
                        console.log('[extractDeviceCode]: read PTX .s file, length', asm.length);
                        console.log('[extractDeviceCode]: PTX .s preview:\n', asm.slice(0, 300));

                        const archAndCode = scaleMatch[1];
                        console.log('[extractDeviceCode]: archAndCode', archAndCode);

                        const nameAndArch = `PTX` + (archAndCode ? ` (${archAndCode.toLowerCase()})` : '');
                        console.log('[extractDeviceCode]: nameAndArch', nameAndArch);

                        Object.assign(devices, {
                            [nameAndArch]: await this.postProcessAsm(
                                {
                                    okToCache: demangle,
                                    ...this.ptxParser.process(asm, {...filters, binary: false}),
                                },
                                {...filters, binary: false},
                            ),
                        });
                        console.log('[extractDeviceCode]: PTX device entry written for', nameAndArch);

                        //
                        // PTX -> CUBIN with ptxas
                        //
                        const cubinPath = Path.join(dirPath, `${Path.basename(name, '.s')}.cubin`);
                        const ptxasCmd = '/product/ubuntu24-x86_64/apps/CUDA/12.8.0/bin/ptxas';
                        const ptxasArgs = ['-arch', archAndCode, name, '-o', cubinPath];
                        console.log('[extractDeviceCode]: running ptxas:', ptxasCmd, ptxasArgs.join(' '));
                        console.log('[extractDeviceCode]: ptxas cwd:', dirPath);

                        try {
                            const ptxasResult = await this.exec(ptxasCmd, ptxasArgs, {customCwd: dirPath});
                            console.log('[extractDeviceCode]: ptxas exit code', ptxasResult.code);
                            console.log('[extractDeviceCode]: ptxas stdout', ptxasResult.stdout);
                            console.log('[extractDeviceCode]: ptxas stderr', ptxasResult.stderr);

                            const filesAfterPtxas = await fs.readdir(dirPath);
                            console.log('[extractDeviceCode]: files after ptxas', filesAfterPtxas);

                            let cubinSize: number | null = null;
                            try {
                                cubinSize = (await fs.stat(cubinPath)).size;
                            } catch {
                                console.warn('[extractDeviceCode]: cubin file does not exist at', cubinPath);
                            }
                            console.log('[extractDeviceCode]: cubin size bytes', cubinSize);

                            if (ptxasResult.code !== 0) {
                                console.warn('[extractDeviceCode]: ptxas failed, skipping SASS');
                                return;
                            }

                            //
                            // CUBIN -> SASS with nvdisasm
                            //
                            const nvdisasmCmd = '/product/ubuntu24-x86_64/apps/CUDA/12.8.0/bin/nvdisasm';
                            const nvdisasmArgs = [cubinPath];
                            console.log('[extractDeviceCode]: running nvdisasm:', nvdisasmCmd, nvdisasmArgs.join(' '));
                            console.log('[extractDeviceCode]: nvdisasm cwd:', dirPath);

                            const {code, stdout} = await this.exec(nvdisasmCmd, nvdisasmArgs, {customCwd: dirPath});
                            console.log('[extractDeviceCode]: nvdisasm exit code', code);
                            console.log('[extractDeviceCode]: nvdisasm stdout length', stdout.length);
                            console.log('[extractDeviceCode]: nvdisasm stdout \n', stdout);

                            const filesAfterNvdisasm = await fs.readdir(dirPath);
                            console.log('[extractDeviceCode]: files after nvdisasm', filesAfterNvdisasm);

                            const sassAsm = code === 0
                                ? this.postProcessObjdumpOutput(stdout)
                                : `<nvdisasm failed with code ${code}>`;
                            console.log('[extractDeviceCode]: sassAsm length after postProcess', sassAsm.length);
                            console.log('[extractDeviceCode]: sassAsm:\n', sassAsm);

                            const sassNameAndArch = `SASS` + (archAndCode ? ` (${archAndCode.toLowerCase()})` : '');
                            console.log('[extractDeviceCode]: writing SASS device entry for', sassNameAndArch);

                            Object.assign(devices, {
                                [sassNameAndArch]: await this.postProcessAsm(
                                    {
                                        okToCache: demangle,
                                        ...this.deviceAsmParser.process(sassAsm, {...filters, binary: false}),
                                    },
                                    {...filters, binary: false},
                                ),
                            });
                            console.log('[extractDeviceCode]: SASS device entry written for', sassNameAndArch);
                            console.log('[extractDeviceCode]: devices keys now', Object.keys(devices));

                        } catch (err) {
                            console.error('[extractDeviceCode]: exception during SASS generation', err);
                        }
                    }),
            );

            console.log('[extractDeviceCode]: final devices keys', Object.keys(devices));
            result.devices = devices;
        }

        return result;
    }
}