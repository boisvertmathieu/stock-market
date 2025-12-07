# n8n Workflow Update Instructions

## Adding Company News to Webhook API

To enable sentiment analysis, you need to add a Company News node to your n8n workflow.

### Step 1: Add HTTP Request Node for Company News

1. Open your n8n workflow "FinnHub API with Webhook Endpoint"
2. Add a new **HTTP Request** node after "FinnHub Recommendations"
3. Configure it as follows:

**Node Name:** `FinnHub Company News`

**Settings:**

- URL: `https://finnhub.io/api/v1/company-news`
- Method: GET
- Authentication: Generic Credential Type → HTTP Header Auth
- Credential: Your existing FinnHub Header Auth

**Query Parameters:**

| Parameter | Value |
|-----------|-------|
| symbol | `={{ $('Webhook Loop Over Items').item.json.ticker }}` |
| from | `={{ DateTime.now().minus({days: 3}).toFormat('yyyy-MM-dd') }}` |
| to | `={{ DateTime.now().toFormat('yyyy-MM-dd') }}` |

### Step 2: Update "Combine Ticker Data" Node

Replace the existing code with this updated version that includes news:

```javascript
// Combine all data for this ticker
const ticker = $('Webhook Loop Over Items').item.json.ticker;
const quote = $('FinnHub Quote').item.json;
const metrics = $('FinnHub Metrics').item.json;
const recommendations = $('FinnHub Recommendations').all();
const newsItems = $('FinnHub Company News').all();

// Get latest recommendation
let latestRec = null;
if (recommendations && recommendations.length > 0 && recommendations[0].json) {
  latestRec = recommendations[0].json;
}

// Process news (limit to 5 most recent)
let news = [];
if (newsItems && newsItems.length > 0) {
  for (let i = 0; i < Math.min(5, newsItems.length); i++) {
    const item = newsItems[i].json;
    if (item && item.headline) {
      news.push({
        headline: item.headline || '',
        summary: item.summary || '',
        datetime: item.datetime || null,
        source: item.source || '',
        url: item.url || ''
      });
    }
  }
}

return {
  ticker: ticker,
  timestamp: new Date().toISOString(),
  quote: {
    current_price: quote.c || null,
    change: quote.d || null,
    change_percent: quote.dp || null,
    high: quote.h || null,
    low: quote.l || null,
    open: quote.o || null,
    previous_close: quote.pc || null
  },
  metrics: {
    pe_ratio: metrics.metric?.peBasicExclExtraTTM || null,
    pe_forward: metrics.metric?.peTTM || null,
    pb_ratio: metrics.metric?.pbQuarterly || null,
    ps_ratio: metrics.metric?.psTTM || null,
    eps: metrics.metric?.epsBasicExclExtraItemsTTM || null,
    eps_growth: metrics.metric?.epsGrowth5Y || null,
    revenue_growth: metrics.metric?.revenueGrowth5Y || null,
    profit_margin: metrics.metric?.netProfitMarginTTM || null,
    gross_margin: metrics.metric?.grossMarginTTM || null,
    roe: metrics.metric?.roeTTM || null,
    roa: metrics.metric?.roaTTM || null,
    debt_equity: metrics.metric?.totalDebtToEquityQuarterly || null,
    current_ratio: metrics.metric?.currentRatioQuarterly || null,
    dividend_yield: metrics.metric?.dividendYieldIndicatedAnnual || null,
    beta: metrics.metric?.beta || null,
    market_cap: metrics.metric?.marketCapitalization || null,
    week52_high: metrics.metric?.['52WeekHigh'] || null,
    week52_low: metrics.metric?.['52WeekLow'] || null,
    avg_volume_10d: metrics.metric?.['10DayAverageTradingVolume'] || null
  },
  analyst: {
    buy: latestRec?.buy || 0,
    hold: latestRec?.hold || 0,
    sell: latestRec?.sell || 0,
    strong_buy: latestRec?.strongBuy || 0,
    strong_sell: latestRec?.strongSell || 0,
    period: latestRec?.period || null
  },
  news: news
};
```

### Step 3: Update Connections

Connect the nodes in this order:

1. FinnHub Recommendations → FinnHub Company News
2. FinnHub Company News → Combine Ticker Data

### API Rate Limits

With 12 tickers × 4 API calls = **48 calls per webhook execution**

Finnhub limits:

- 60 calls/minute (free tier)
- 30 calls/second max

You're safely within limits!
