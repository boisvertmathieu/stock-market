import React, { useState, useEffect, useCallback } from 'react';
import { render, Box, Text, useInput, useApp, Spacer } from 'ink';
import Spinner from 'ink-spinner';
import TextInput from 'ink-text-input';
import { spawn } from 'child_process';

// Types
interface TickerResult {
    ticker: string;
    name: string;
    price: number;
    change_percent: number;
    technical_signal: string;
    ml_prediction: string;
    overall_action: string;
    action_strength: number;
    key_factors: string[];
    entry_price?: number;
    stop_loss?: number;
    take_profit?: number;
}

interface MarketSummary {
    bullish: number;
    bearish: number;
    neutral: number;
    avg_change_percent: number;
}

interface AppState {
    view: 'dashboard' | 'list' | 'detail' | 'search';
    loading: boolean;
    error: string | null;
    results: TickerResult[];
    selectedIndex: number;
    selectedTicker: TickerResult | null;
    summary: MarketSummary | null;
    searchQuery: string;
}

// Color helpers
const getActionColor = (action: string): string => {
    if (action.includes('BUY')) return 'green';
    if (action.includes('SELL')) return 'red';
    return 'gray';
};

const getChangeColor = (change: number): string => {
    return change >= 0 ? 'green' : 'red';
};

// Header component
const Header: React.FC<{ title: string }> = ({ title }) => (
    <Box borderStyle="round" borderColor="cyan" paddingX={2} marginBottom={1}>
        <Text bold color="cyan">📊 {title}</Text>
        <Spacer />
        <Text dimColor>Press ? for help</Text>
    </Box>
);

// Status bar
const StatusBar: React.FC<{ summary: MarketSummary | null }> = ({ summary }) => (
    <Box borderStyle="single" borderColor="gray" paddingX={1} marginTop={1}>
        {summary ? (
            <>
                <Text color="green">▲ {summary.bullish}</Text>
                <Text> | </Text>
                <Text color="red">▼ {summary.bearish}</Text>
                <Text> | </Text>
                <Text dimColor>― {summary.neutral}</Text>
                <Text> | </Text>
                <Text>Avg: </Text>
                <Text color={getChangeColor(summary.avg_change_percent)}>
                    {summary.avg_change_percent >= 0 ? '+' : ''}{summary.avg_change_percent.toFixed(2)}%
                </Text>
            </>
        ) : (
            <Text dimColor>Loading market data...</Text>
        )}
    </Box>
);

// Ticker row
const TickerRow: React.FC<{
    result: TickerResult;
    selected: boolean;
    compact?: boolean;
}> = ({ result, selected, compact = false }) => {
    const bgColor = selected ? 'blue' : undefined;

    return (
        <Box backgroundColor={bgColor} paddingX={1}>
            <Box width={8}>
                <Text bold color={selected ? 'white' : 'cyan'}>{result.ticker}</Text>
            </Box>
            {!compact && (
                <Box width={20}>
                    <Text dimColor>{result.name?.slice(0, 18) || ''}</Text>
                </Box>
            )}
            <Box width={12}>
                <Text>${result.price?.toFixed(2) || '0.00'}</Text>
            </Box>
            <Box width={10}>
                <Text color={getChangeColor(result.change_percent || 0)}>
                    {(result.change_percent || 0) >= 0 ? '+' : ''}{(result.change_percent || 0).toFixed(2)}%
                </Text>
            </Box>
            <Box width={12}>
                <Text>{result.technical_signal || 'N/A'}</Text>
            </Box>
            <Box width={12}>
                <Text color={result.ml_prediction?.includes('UP') ? 'green' : result.ml_prediction?.includes('DOWN') ? 'red' : 'gray'}>
                    {result.ml_prediction || 'N/A'}
                </Text>
            </Box>
            <Box>
                <Text bold color={getActionColor(result.overall_action || 'HOLD')}>
                    {result.overall_action || 'HOLD'}
                </Text>
            </Box>
        </Box>
    );
};

// Ticker list view
const TickerListView: React.FC<{
    results: TickerResult[];
    selectedIndex: number;
    maxRows?: number;
}> = ({ results, selectedIndex, maxRows = 15 }) => {
    // Calculate visible range
    const startIdx = Math.max(0, selectedIndex - Math.floor(maxRows / 2));
    const endIdx = Math.min(results.length, startIdx + maxRows);
    const visibleResults = results.slice(startIdx, endIdx);

    return (
        <Box flexDirection="column">
            {/* Header row */}
            <Box paddingX={1} marginBottom={1}>
                <Box width={8}><Text bold dimColor>TICKER</Text></Box>
                <Box width={20}><Text bold dimColor>NAME</Text></Box>
                <Box width={12}><Text bold dimColor>PRICE</Text></Box>
                <Box width={10}><Text bold dimColor>CHANGE</Text></Box>
                <Box width={12}><Text bold dimColor>SIGNAL</Text></Box>
                <Box width={12}><Text bold dimColor>ML</Text></Box>
                <Box><Text bold dimColor>ACTION</Text></Box>
            </Box>

            {/* Ticker rows */}
            {visibleResults.map((result, idx) => (
                <TickerRow
                    key={result.ticker}
                    result={result}
                    selected={startIdx + idx === selectedIndex}
                />
            ))}

            {/* Scroll indicator */}
            <Box marginTop={1}>
                <Text dimColor>
                    [{selectedIndex + 1}/{results.length}] ↑↓ Navigate | Enter: Details | q: Quit
                </Text>
            </Box>
        </Box>
    );
};

// Detail view
const DetailView: React.FC<{ result: TickerResult; onBack: () => void }> = ({ result, onBack }) => (
    <Box flexDirection="column" padding={1}>
        <Box borderStyle="round" borderColor="cyan" paddingX={2} paddingY={1}>
            <Text bold color="white">{result.name || result.ticker}</Text>
            <Text> </Text>
            <Text dimColor>({result.ticker})</Text>
        </Box>

        <Box marginY={1}>
            <Box flexDirection="column" width="50%">
                <Text bold color="cyan">💰 PRICE</Text>
                <Text>  Current: <Text bold>${result.price?.toFixed(2) || '0.00'}</Text></Text>
                <Text>  Change: <Text color={getChangeColor(result.change_percent || 0)}>
                    {(result.change_percent || 0) >= 0 ? '+' : ''}{(result.change_percent || 0).toFixed(2)}%
                </Text></Text>
            </Box>

            <Box flexDirection="column" width="50%">
                <Text bold color="cyan">📈 ANALYSIS</Text>
                <Text>  Technical: {result.technical_signal || 'N/A'}</Text>
                <Text>  ML Prediction: <Text color={result.ml_prediction?.includes('UP') ? 'green' : result.ml_prediction?.includes('DOWN') ? 'red' : 'white'}>
                    {result.ml_prediction || 'N/A'}
                </Text></Text>
            </Box>
        </Box>

        <Box borderStyle="round" borderColor={getActionColor(result.overall_action || 'HOLD')} paddingX={2} paddingY={1} marginY={1}>
            <Text bold color={getActionColor(result.overall_action || 'HOLD')}>
                🎯 RECOMMENDATION: {result.overall_action || 'HOLD'}
            </Text>
        </Box>

        {result.key_factors && result.key_factors.length > 0 && (
            <Box flexDirection="column" marginY={1}>
                <Text bold color="cyan">📋 KEY FACTORS</Text>
                {result.key_factors.map((factor, idx) => (
                    <Text key={idx}>  • {factor}</Text>
                ))}
            </Box>
        )}

        {result.entry_price && (
            <Box flexDirection="column" marginY={1}>
                <Text bold color="cyan">📍 TRADE LEVELS</Text>
                <Text>  Entry:       <Text bold>${result.entry_price?.toFixed(2)}</Text></Text>
                <Text>  Stop Loss:   <Text color="red">${result.stop_loss?.toFixed(2)}</Text></Text>
                <Text>  Take Profit: <Text color="green">${result.take_profit?.toFixed(2)}</Text></Text>
            </Box>
        )}

        <Box marginTop={2}>
            <Text dimColor>Press Esc or Backspace to go back</Text>
        </Box>
    </Box>
);

// Help view
const HelpView: React.FC = () => (
    <Box flexDirection="column" padding={1}>
        <Text bold color="cyan">⌨️ KEYBOARD SHORTCUTS</Text>
        <Text></Text>
        <Text>  <Text bold>↑/↓</Text>     Navigate list</Text>
        <Text>  <Text bold>Enter</Text>   View ticker details</Text>
        <Text>  <Text bold>Esc</Text>     Go back</Text>
        <Text>  <Text bold>/</Text>       Search ticker</Text>
        <Text>  <Text bold>r</Text>       Refresh data</Text>
        <Text>  <Text bold>b</Text>       Show only BUY signals</Text>
        <Text>  <Text bold>s</Text>       Show only SELL signals</Text>
        <Text>  <Text bold>a</Text>       Show all</Text>
        <Text>  <Text bold>?</Text>       Toggle help</Text>
        <Text>  <Text bold>q</Text>       Quit</Text>
        <Text></Text>
        <Text dimColor>Press any key to close help</Text>
    </Box>
);

// Loading view
const LoadingView: React.FC<{ message: string }> = ({ message }) => (
    <Box padding={2}>
        <Text color="cyan">
            <Spinner type="dots" />
        </Text>
        <Text> {message}</Text>
    </Box>
);

// Main App
const App: React.FC = () => {
    const { exit } = useApp();

    const [state, setState] = useState<AppState>({
        view: 'list',
        loading: true,
        error: null,
        results: [],
        selectedIndex: 0,
        selectedTicker: null,
        summary: null,
        searchQuery: '',
    });

    const [showHelp, setShowHelp] = useState(false);
    const [filter, setFilter] = useState<'all' | 'buy' | 'sell'>('all');

    // Filter results
    const filteredResults = state.results.filter(r => {
        if (filter === 'buy') return r.action_strength > 0;
        if (filter === 'sell') return r.action_strength < 0;
        return true;
    });

    // Fetch data from Python backend
    const fetchData = useCallback(async () => {
        setState(s => ({ ...s, loading: true, error: null }));

        try {
            // Spawn Python process
            const pythonProcess = spawn('python', ['-m', 'src.main', 'scan', '--quick', '--json'], {
                cwd: process.cwd().replace('/tui', ''),
                env: { ...process.env, PYTHONPATH: process.cwd().replace('/tui', '') },
            });

            let output = '';
            let errorOutput = '';

            pythonProcess.stdout.on('data', (data) => {
                output += data.toString();
            });

            pythonProcess.stderr.on('data', (data) => {
                errorOutput += data.toString();
            });

            pythonProcess.on('close', (code) => {
                if (code !== 0) {
                    setState(s => ({ ...s, loading: false, error: `Python exited with code ${code}: ${errorOutput}` }));
                    return;
                }

                try {
                    const data = JSON.parse(output);
                    setState(s => ({
                        ...s,
                        loading: false,
                        results: data.results || [],
                        summary: data.summary?.market_sentiment || null,
                    }));
                } catch (e) {
                    setState(s => ({ ...s, loading: false, error: `Failed to parse response: ${e}` }));
                }
            });
        } catch (error) {
            setState(s => ({ ...s, loading: false, error: `Failed to fetch data: ${error}` }));
        }
    }, []);

    // Initial fetch
    useEffect(() => {
        fetchData();
    }, [fetchData]);

    // Keyboard input handling
    useInput((input, key) => {
        if (showHelp) {
            setShowHelp(false);
            return;
        }

        if (state.view === 'search') {
            if (key.escape) {
                setState(s => ({ ...s, view: 'list', searchQuery: '' }));
            }
            return;
        }

        if (state.view === 'detail') {
            if (key.escape || key.backspace) {
                setState(s => ({ ...s, view: 'list', selectedTicker: null }));
            }
            return;
        }

        // List view controls
        if (input === 'q') {
            exit();
            return;
        }

        if (input === '?') {
            setShowHelp(true);
            return;
        }

        if (input === 'r') {
            fetchData();
            return;
        }

        if (input === 'b') {
            setFilter('buy');
            setState(s => ({ ...s, selectedIndex: 0 }));
            return;
        }

        if (input === 's') {
            setFilter('sell');
            setState(s => ({ ...s, selectedIndex: 0 }));
            return;
        }

        if (input === 'a') {
            setFilter('all');
            setState(s => ({ ...s, selectedIndex: 0 }));
            return;
        }

        if (input === '/') {
            setState(s => ({ ...s, view: 'search' }));
            return;
        }

        if (key.upArrow) {
            setState(s => ({
                ...s,
                selectedIndex: Math.max(0, s.selectedIndex - 1)
            }));
            return;
        }

        if (key.downArrow) {
            setState(s => ({
                ...s,
                selectedIndex: Math.min(filteredResults.length - 1, s.selectedIndex + 1)
            }));
            return;
        }

        if (key.return && filteredResults.length > 0) {
            setState(s => ({
                ...s,
                view: 'detail',
                selectedTicker: filteredResults[s.selectedIndex]
            }));
            return;
        }
    });

    // Search handler
    const handleSearchChange = (value: string) => {
        setState(s => ({ ...s, searchQuery: value }));
    };

    const handleSearchSubmit = () => {
        const query = state.searchQuery.toUpperCase();
        const idx = filteredResults.findIndex(r => r.ticker === query);
        if (idx >= 0) {
            setState(s => ({ ...s, view: 'list', selectedIndex: idx, searchQuery: '' }));
        } else {
            setState(s => ({ ...s, view: 'list', searchQuery: '' }));
        }
    };

    // Render
    return (
        <Box flexDirection="column" height={process.stdout.rows - 2}>
            <Header title="Stock Market Scanner" />

            {showHelp ? (
                <HelpView />
            ) : state.loading ? (
                <LoadingView message="Scanning market... This may take a few minutes." />
            ) : state.error ? (
                <Box padding={2}>
                    <Text color="red">Error: {state.error}</Text>
                </Box>
            ) : state.view === 'search' ? (
                <Box padding={1}>
                    <Text>Search ticker: </Text>
                    <TextInput
                        value={state.searchQuery}
                        onChange={handleSearchChange}
                        onSubmit={handleSearchSubmit}
                    />
                </Box>
            ) : state.view === 'detail' && state.selectedTicker ? (
                <DetailView
                    result={state.selectedTicker}
                    onBack={() => setState(s => ({ ...s, view: 'list', selectedTicker: null }))}
                />
            ) : (
                <>
                    <Box marginBottom={1}>
                        <Text dimColor>Filter: </Text>
                        <Text bold color={filter === 'all' ? 'cyan' : 'gray'}>[a]ll </Text>
                        <Text bold color={filter === 'buy' ? 'green' : 'gray'}>[b]uy </Text>
                        <Text bold color={filter === 'sell' ? 'red' : 'gray'}>[s]ell </Text>
                        <Spacer />
                        <Text dimColor>Showing {filteredResults.length} results</Text>
                    </Box>
                    <TickerListView
                        results={filteredResults}
                        selectedIndex={state.selectedIndex}
                    />
                </>
            )}

            <Spacer />
            <StatusBar summary={state.summary} />
        </Box>
    );
};

// Render the app
render(<App />);
