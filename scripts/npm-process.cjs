'use strict';

// Windows npm.cmd is a batch wrapper, not a native executable. Execute its
// JavaScript entry point with Node so no user argument crosses a shell parser.
const cp = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

function npmCommand(args = []) {
  if (!Array.isArray(args) || args.some(arg => typeof arg !== 'string')) {
    throw new TypeError('npm arguments must be an array of strings');
  }
  if (process.platform !== 'win32') return { file: 'npm', args: [...args] };

  const cli = path.join(path.dirname(process.execPath), 'node_modules', 'npm', 'bin', 'npm-cli.js');
  if (!fs.existsSync(cli)) {
    throw new Error(`npm CLI not found beside the active Node.js: ${cli}. Use the Node.js installation that includes npm.`);
  }
  return { file: process.execPath, args: [cli, ...args] };
}

function nativeOptions(options = {}) {
  if (options.shell) throw new TypeError('npm-process requires shell: false');
  return { windowsHide: true, ...options, shell: false, windowsVerbatimArguments: false };
}

function npmSpawnSync(args, options) {
  const command = npmCommand(args);
  return cp.spawnSync(command.file, command.args, nativeOptions(options));
}

function npmSpawn(args, options) {
  const command = npmCommand(args);
  return cp.spawn(command.file, command.args, nativeOptions(options));
}

function npmExecFile(args, options, callback) {
  if (typeof options === 'function') {
    callback = options;
    options = undefined;
  }
  const command = npmCommand(args);
  return cp.execFile(command.file, command.args, nativeOptions(options), callback);
}

module.exports = { npmCommand, npmSpawnSync, npmSpawn, npmExecFile };

if (require.main === module) {
  try {
    const result = npmSpawnSync(process.argv.slice(2), { stdio: 'inherit' });
    if (result.error) throw result.error;
    process.exitCode = result.status ?? 1;
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
