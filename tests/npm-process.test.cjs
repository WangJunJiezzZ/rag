'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { test, after } = require('node:test');
const { npmSpawnSync, npmSpawn, npmExecFile } = require('../scripts/npm-process.cjs');

const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'rag-npm-test-'));
const options = {
  encoding: 'utf8', timeout: 15000,
  env: { ...process.env, npm_config_cache: path.join(temp, 'cache'), NO_UPDATE_NOTIFIER: '1' },
};
after(() => fs.rmSync(temp, { recursive: true, force: true }));

function spawnText(args) {
  return new Promise((resolve, reject) => {
    const child = npmSpawn(args, options);
    let stdout = '', stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', chunk => { stdout += chunk; });
    child.stderr.on('data', chunk => { stderr += chunk; });
    child.on('error', reject);
    child.on('close', status => resolve({ status, stdout, stderr }));
  });
}

function execText(args) {
  return new Promise((resolve, reject) => {
    npmExecFile(args, options, (error, stdout, stderr) => {
      if (error) reject(error);
      else resolve({ status: 0, stdout, stderr });
    });
  });
}

for (const [name, invoke] of [
  ['spawnSync', args => npmSpawnSync(args, options)],
  ['spawn', spawnText], ['execFile', execText],
]) {
  test(`${name}: npm starts successfully`, async () => {
    const result = await invoke(['--version']);
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout.trim(), /^\d+\.\d+\.\d+/);
  });

  test(`${name}: spaces, Unicode and shell metacharacters stay literal`, async () => {
    // config get is read-only; this invalid-looking path is just a string.
    const prefix = path.join(temp, '空 格 & echo injected | < > ^ %COMSPEC% !name! (x) "quote"');
    const result = await invoke(['--prefix', prefix, 'config', 'get', 'prefix']);
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout.trim(), prefix);
    assert.equal(fs.existsSync(path.join(temp, 'injected')), false);
  });
}

test('npm failures retain their nonzero exit status', () => {
  const result = npmSpawnSync(['rag-nonexistent-command'], options);
  assert.equal(result.error, undefined);
  assert.notEqual(result.status, 0);
});

test('shell execution and command strings are rejected', () => {
  assert.throws(() => npmSpawnSync('--version', options), /array of strings/);
  assert.throws(() => npmSpawnSync(['--version'], { shell: true }), /shell: false/);
});
