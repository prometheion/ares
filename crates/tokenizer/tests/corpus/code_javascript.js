'use strict';
const express = require('express');
const app = express();

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function fetchAll(urls, { retries = 3, delay = 250 } = {}) {
  const results = [];
  for (const url of urls) {
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
        if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
        results.push(await res.json());
        break;
      } catch (err) {
        if (attempt === retries) throw err;
        await sleep(delay * 2 ** attempt);
      }
    }
  }
  return results;
}

app.get('/api/items/:id', (req, res) => {
  const id = Number.parseInt(req.params.id, 10);
  if (!Number.isFinite(id) || id < 0) return res.status(400).json({ error: 'bad id' });
  res.json({ id, name: `item-${id}`, tags: ['a', 'b'], price: 19.99, inStock: id % 2 === 0 });
});

const re = /^(?<year>\d{4})-(?<month>\d{2})-(?<day>\d{2})$/u;
console.log(re.exec('2024-02-29')?.groups ?? null, [1, 2, 3].map((x) => x ** 2).join(', '));
app.listen(process.env.PORT || 3000, () => console.log('listening'));
