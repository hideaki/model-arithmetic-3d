// Collects the files a static host needs into one folder, for publishing the pages
// (GitHub Pages: .github/workflows/pages.yml). The project itself has no build step;
// this only copies, with two differences from the project folder:
//
//   index.html     is models.html, the list of models. The hand-written page that
//                  is index.html in the project is not published.
//   compiled.html  has no question panel. A static host cannot answer questions
//                  (they need serve.py and an API key), and the runtime leaves the
//                  panel alone when the page has none.
//
//   node tools/build-site.mjs [folder]      default: _site
//
// To look at the result:  python3 -m http.server 5176 --directory _site

import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const MARK = '.built-by-build-site'; // only a folder that holds this file is ever emptied

// Copied as they are. Everything compiled.html loads is in the first four.
const COPIED = ['lang', 'runtime', 'programs', 'src/style.css', 'docs/images/thumbs', 'LICENSE'];

const ASK_PANEL = /\s*(<!--[^>]*-->\s*)?<section id="ask">[\s\S]*?<\/section>/g;

export function withoutAskPanel(html) {
  const found = html.match(ASK_PANEL)?.length ?? 0;
  if (found !== 1) throw new Error(`compiled.html: expected one <section id="ask">, found ${found}`);
  return html.replace(ASK_PANEL, '');
}

export function buildSite(out) {
  out = resolve(out);
  if (existsSync(out)) {
    if (!existsSync(join(out, MARK)) && readdirSync(out).length) throw new Error(`${out} exists and was not made by this script; choose another folder or remove it yourself`);
    rmSync(out, { recursive: true });
  }
  mkdirSync(out, { recursive: true });
  writeFileSync(join(out, MARK), 'Made by tools/build-site.mjs. Safe to delete.\n');
  for (const path of COPIED) cpSync(join(ROOT, path), join(out, path), { recursive: true });
  writeFileSync(join(out, 'index.html'), readFileSync(join(ROOT, 'models.html'), 'utf8'));
  writeFileSync(join(out, 'compiled.html'), withoutAskPanel(readFileSync(join(ROOT, 'compiled.html'), 'utf8')));
  return out;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const out = buildSite(process.argv[2] || join(ROOT, '_site'));
  const count = (dir) => readdirSync(dir, { withFileTypes: true }).reduce((n, e) => n + (e.isDirectory() ? count(join(dir, e.name)) : 1), 0);
  console.log(`${out}: ${count(out) - 1} files`);
}
