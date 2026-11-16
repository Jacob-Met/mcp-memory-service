'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const staticRoot = path.join(__dirname, '../../src/mcp_memory_service/web/static');
const css = fs.readFileSync(path.join(staticRoot, 'style.css'), 'utf8');
const html = fs.readFileSync(path.join(staticRoot, 'index.html'), 'utf8');

test('all dashboard destinations remain keyboard-accessible buttons', () => {
    const nav = html.match(/<nav class="main-nav">([\s\S]*?)<\/nav>/)?.[1];
    assert.ok(nav);
    const destinations = [...nav.matchAll(/<button class="nav-item(?: active)?" data-view="([^"]+)"/g)]
        .map((match) => match[1]);
    assert.deepEqual(destinations, [
        'dashboard', 'search', 'browse', 'documents', 'manage',
        'analytics', 'qualityAnalytics', 'apiDocs',
    ]);
});

test('narrow navigation remains one scrollable row without widening the grid page', () => {
    const start = css.indexOf('@media (max-width: 1100px) {');
    const end = css.indexOf('/* Responsive Design */', start);
    assert.ok(start >= 0 && end > start);
    const rules = css.slice(start, end);
    const container = rules.match(/\.nav-container\s*\{([^}]+)\}/)?.[1];
    const main = rules.match(/\.main-nav\s*\{([^}]+)\}/)?.[1];
    const item = rules.match(/\.nav-item\s*\{([^}]+)\}/)?.[1];
    assert.ok(container && main && item);
    assert.match(main, /min-width:\s*0\s*;/);
    assert.match(main, /overflow:\s*hidden\s*;/);
    assert.match(container, /flex-wrap:\s*nowrap\s*;/);
    assert.match(container, /overflow-x:\s*auto\s*;/);
    assert.match(item, /flex:\s*none\s*;/);
    assert.match(item, /white-space:\s*nowrap\s*;/);
});
