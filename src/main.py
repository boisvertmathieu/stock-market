#!/usr/bin/env python3
"""
Stock Market Analysis CLI
Main entry point for the application.
"""

import argparse
import json
import sys
import logging
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

console = Console()


def create_analysis_table(results: list) -> Table:
    """Create a rich table from analysis results."""
    table = Table(title="📊 Market Analysis Results", show_header=True, header_style="bold cyan")
    
    table.add_column("Ticker", style="bold white", width=8)
    table.add_column("Name", width=20)
    table.add_column("Price", justify="right", style="white")
    table.add_column("Change", justify="right")
    table.add_column("Signal", justify="center")
    table.add_column("ML", justify="center")
    table.add_column("Action", justify="center")
    
    for r in results:
        # Color for change
        change_style = "green" if r.get('change_percent', 0) >= 0 else "red"
        change_text = f"{r.get('change_percent', 0):+.2f}%"
        
        # Color for action
        action = r.get('overall_action', 'HOLD')
        if 'BUY' in action:
            action_style = "bold green"
        elif 'SELL' in action:
            action_style = "bold red"
        else:
            action_style = "dim"
        
        # ML confidence coloring
        ml_pred = r.get('ml_prediction', 'N/A')
        if 'UP' in ml_pred:
            ml_style = "green"
        elif 'DOWN' in ml_pred:
            ml_style = "red"
        else:
            ml_style = "dim"
        
        table.add_row(
            r.get('ticker', ''),
            r.get('name', '')[:20],
            f"${r.get('price', 0):.2f}",
            Text(change_text, style=change_style),
            r.get('technical_signal', 'N/A'),
            Text(ml_pred, style=ml_style),
            Text(action, style=action_style),
        )
    
    return table


def create_detail_panel(analysis: dict) -> Panel:
    """Create a detailed analysis panel for a single ticker."""
    ticker = analysis.get('ticker', 'N/A')
    info = analysis.get('info', {})
    price = analysis.get('price', {})
    tech = analysis.get('technical_signal', {})
    ml = analysis.get('ml_prediction', {})
    suggestion = analysis.get('trade_suggestion', {})
    
    # Build content
    lines = []
    
    # Header
    change_color = "green" if price.get('change_percent', 0) >= 0 else "red"
    lines.append(f"[bold white]{info.get('name', ticker)}[/] ({ticker})")
    lines.append(f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}")
    lines.append("")
    
    # Price section
    lines.append("[bold cyan]💰 PRICE[/]")
    lines.append(f"  Current: [bold]${price.get('current', 0):.2f}[/]")
    lines.append(f"  Change:  [{change_color}]{price.get('change', 0):+.2f} ({price.get('change_percent', 0):+.2f}%)[/{change_color}]")
    lines.append("")
    
    # Technical signals
    lines.append("[bold cyan]📈 TECHNICAL ANALYSIS[/]")
    lines.append(f"  Overall: [bold]{tech.get('signal', 'N/A')}[/] (Score: {tech.get('score', 0):.2f})")
    lines.append(f"  Action:  {tech.get('action', 'N/A')}")
    lines.append("")
    
    # Individual indicators
    indicators = analysis.get('indicators', {})
    for name, ind in list(indicators.items())[:4]:
        signal_color = "green" if ind.get('signal_value', 0) > 0 else "red" if ind.get('signal_value', 0) < 0 else "white"
        lines.append(f"  • {name.upper()}: [{signal_color}]{ind.get('signal')}[/{signal_color}] - {ind.get('description', '')[:50]}")
    lines.append("")
    
    # ML Prediction
    lines.append("[bold cyan]🤖 ML PREDICTION[/]")
    ml_pred = ml.get('prediction') or 'N/A'
    ml_color = "green" if ml_pred and "UP" in str(ml_pred) else "red" if ml_pred and "DOWN" in str(ml_pred) else "white"
    lines.append(f"  Prediction:  [{ml_color}]{ml_pred}[/{ml_color}]")
    lines.append(f"  Confidence:  {ml.get('confidence', 'N/A')} ({ml.get('probability', 0):.0%})")
    explanation = ml.get('explanation') or 'N/A'
    lines.append(f"  Explanation: {str(explanation)[:80]}")
    lines.append("")
    
    # Trade suggestion
    if suggestion.get('action') not in [None, 'HOLD']:
        action_color = "green" if suggestion.get('action') == 'BUY' else "red"
        lines.append(f"[bold cyan]🎯 TRADE SUGGESTION[/]")
        lines.append(f"  Action:      [bold {action_color}]{suggestion.get('action')}[/bold {action_color}]")
        lines.append(f"  Entry:       ${suggestion.get('entry', 0):.2f}")
        lines.append(f"  Stop Loss:   ${suggestion.get('stop_loss', 0):.2f}")
        lines.append(f"  Take Profit: ${suggestion.get('take_profit', 0):.2f}")
    else:
        lines.append("[bold yellow]⏸️  HOLD - No trade suggested[/]")
    
    return Panel("\n".join(lines), title=f"📊 {ticker} Analysis", border_style="cyan")


def cmd_analyze(args):
    """Analyze a single ticker."""
    from src.api import StockAPI
    
    api = StockAPI()
    
    with console.status(f"[bold cyan]Analyzing {args.ticker}..."):
        result = api.handle_command({
            'action': 'analyze',
            'params': {'ticker': args.ticker, 'period': args.period}
        })
    
    if 'error' in result:
        console.print(f"[red]Error: {result['error']}[/]")
        return 1
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        console.print(create_detail_panel(result))
    
    return 0


def cmd_scan(args):
    """Scan multiple tickers."""
    from src.api import StockAPI
    from src.data_fetcher import TOP_100_TICKERS
    
    # Suppress logging in JSON mode
    if args.json:
        logging.getLogger().setLevel(logging.CRITICAL)
    
    api = StockAPI()
    
    # Determine tickers to scan
    if args.quick:
        tickers = TOP_100_TICKERS[:20]
        if not args.json:
            console.print("[cyan]Running quick scan (top 20 tickers)...[/]")
    else:
        tickers = TOP_100_TICKERS[:args.limit] if args.limit else TOP_100_TICKERS
        if not args.json:
            console.print(f"[cyan]Scanning {len(tickers)} tickers...[/]")
    
    # Progress bar (only if not JSON output)
    if not args.json:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning market...", total=len(tickers))
            
            def update_progress(completed, total, ticker):
                progress.update(task, completed=completed, description=f"Analyzing {ticker}...")
            
            api.scanner.scan_tickers(tickers=tickers, period=args.period, progress_callback=update_progress)
    else:
        # Silent scan for JSON output
        api.scanner.scan_tickers(tickers=tickers, period=args.period)
    
    # Get results
    result = api.handle_command({
        'action': 'quick_scan' if args.quick else 'scan',
        'params': {'tickers': tickers, 'period': args.period}
    })
    
    if 'error' in result:
        console.print(f"[red]Error: {result['error']}[/]")
        return 1
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # Show table
        console.print()
        console.print(create_analysis_table(result.get('results', [])[:30]))
        
        # Show summary
        summary = result.get('summary', {})
        sentiment = summary.get('market_sentiment', {})
        
        console.print()
        console.print(Panel(
            f"[green]Bullish:[/] {sentiment.get('bullish', 0)} | "
            f"[red]Bearish:[/] {sentiment.get('bearish', 0)} | "
            f"[dim]Neutral:[/] {sentiment.get('neutral', 0)} | "
            f"Avg Change: {summary.get('avg_change_percent', 0):+.2f}%",
            title="📊 Market Summary"
        ))
        
        # Show top opportunities
        if 'top_buys' in result and result['top_buys']:
            console.print()
            console.print("[bold green]🚀 Top Buy Opportunities:[/]")
            for r in result['top_buys'][:5]:
                console.print(f"  • [bold]{r['ticker']}[/] @ ${r['price']:.2f} - {', '.join(r.get('key_factors', [])[:2])}")
        
        if 'top_sells' in result and result['top_sells']:
            console.print()
            console.print("[bold red]📉 Top Sell Signals:[/]")
            for r in result['top_sells'][:5]:
                console.print(f"  • [bold]{r['ticker']}[/] @ ${r['price']:.2f} - {', '.join(r.get('key_factors', [])[:2])}")
    
    return 0


def cmd_backtest(args):
    """Run backtesting."""
    from src.api import StockAPI
    
    api = StockAPI()
    
    with console.status(f"[bold cyan]Running backtest for {args.ticker}..."):
        result = api.handle_command({
            'action': 'backtest',
            'params': {
                'ticker': args.ticker,
                'period': args.period,
                'capital': args.capital
            }
        })
    
    if 'error' in result:
        console.print(f"[red]Error: {result['error']}[/]")
        return 1
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        metrics = result.get('metrics', {})
        
        # Color based on performance
        return_color = "green" if metrics.get('total_return', 0) >= 0 else "red"
        sharpe_color = "green" if metrics.get('sharpe_ratio', 0) >= 1 else "yellow" if metrics.get('sharpe_ratio', 0) >= 0 else "red"
        
        table = Table(title=f"📈 Backtest Results: {args.ticker}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("Total Return", f"[{return_color}]{metrics.get('total_return', 0):.2%}[/{return_color}]")
        table.add_row("Annualized Return", f"[{return_color}]{metrics.get('annualized_return', 0):.2%}[/{return_color}]")
        table.add_row("Sharpe Ratio", f"[{sharpe_color}]{metrics.get('sharpe_ratio', 0):.2f}[/{sharpe_color}]")
        table.add_row("Max Drawdown", f"[red]{metrics.get('max_drawdown', 0):.2%}[/red]")
        table.add_row("Win Rate", f"{metrics.get('win_rate', 0):.2%}")
        table.add_row("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
        table.add_row("Total Trades", str(metrics.get('total_trades', 0)))
        table.add_row("Avg Trade P&L", f"${metrics.get('avg_trade_pnl', 0):.2f}")
        
        console.print(table)
    
    return 0


def cmd_simulate(args):
    """Run trading simulation over historical data."""
    from src.data_fetcher import TOP_100_TICKERS
    
    # Suppress logging in JSON mode
    if args.json:
        logging.getLogger().setLevel(logging.CRITICAL)
    
    # Select tickers
    tickers = TOP_100_TICKERS[:args.tickers]
    
    # Use momentum strategy (recommended - beats S&P 500)
    if getattr(args, 'momentum', True):
        from src.momentum_strategy import MomentumStrategy
        
        if not args.json:
            console.print(f"[bold cyan]🚀 Starting Momentum Buy & Hold Simulation[/]")
            console.print(f"Capital: ${args.capital:,.0f} | Period: {args.period} | Universe: {len(tickers)} tickers")
            console.print(f"Strategy: Monthly rebalance, Top 10 momentum stocks")
            console.print()
        
        strategy = MomentumStrategy(
            initial_capital=args.capital,
            top_n=10,
            rebalance_freq=21,  # Monthly
            silent=True
        )
        
        if not args.json:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task = progress.add_task("Simulating...", total=500)
                
                def update_progress(day, total, info):
                    progress.update(task, completed=day, total=total,
                                  description=f"Day {day}/{total}: ${info['portfolio_value']:,.0f}")
                
                result = strategy.run(tickers=tickers, period=args.period, progress_callback=update_progress)
        else:
            result = strategy.run(tickers=tickers, period=args.period)
        
        if args.json:
            output = {
                'strategy': 'momentum',
                'initial_capital': float(result.initial_capital),
                'final_value': float(result.final_value),
                'total_return': float(result.total_return),
                'annualized_return': float(result.annualized_return),
                'sharpe_ratio': float(result.sharpe_ratio),
                'max_drawdown': float(result.max_drawdown),
                'benchmark_return': float(result.benchmark_return),
                'alpha': float(result.alpha),
                'beat_benchmark': bool(result.total_return > result.benchmark_return),
                'holdings': result.portfolio[-1]['holdings'] if result.portfolio else []
            }
            print(json.dumps(output, indent=2))
        else:
            console.print()
            console.print(strategy.format_results(result))
            
            if result.total_return > result.benchmark_return:
                console.print(Panel(
                    f"[bold green]✅ STRATEGY BEAT S&P 500![/]\n\n"
                    f"Strategy Return: [green]{result.total_return:+.2%}[/]\n"
                    f"S&P 500 Return:  [yellow]{result.benchmark_return:+.2%}[/]\n"
                    f"Alpha:           [green]{result.alpha:+.2%}[/]",
                    title="🏆 SIMULATION RESULT",
                    border_style="green"
                ))
            else:
                console.print(Panel(
                    f"[bold red]❌ STRATEGY UNDERPERFORMED S&P 500[/]\n\n"
                    f"Strategy Return: [red]{result.total_return:+.2%}[/]\n"
                    f"S&P 500 Return:  [green]{result.benchmark_return:+.2%}[/]\n"
                    f"Alpha:           [red]{result.alpha:+.2%}[/]",
                    title="📉 SIMULATION RESULT",
                    border_style="red"
                ))
    
    else:
        # Active trading strategy (original)
        from src.enhanced_simulator import EnhancedSimulator
        
        if not args.json:
            console.print(f"[bold cyan]🚀 Starting Active Trading Simulation[/]")
            console.print(f"Capital: ${args.capital:,.0f} | Period: {args.period} | Tickers: {len(tickers)}")
            console.print(f"Max Positions: {args.max_positions} | Position Size: {args.position_size:.0%}")
            console.print()
        
        simulator = EnhancedSimulator(
            initial_capital=args.capital,
            max_positions=args.max_positions,
            base_position_size=args.position_size,
            signal_threshold=0.3,
            use_trailing_stop=True,
            silent=True
        )
        
        if not args.json:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task = progress.add_task("Simulating...", total=500)
                
                def update_progress(day, total, info):
                    progress.update(task, completed=day, total=total,
                                  description=f"Day {day}/{total}: ${info['portfolio_value']:,.0f}")
                
                result = simulator.run(tickers=tickers, period=args.period, progress_callback=update_progress)
        else:
            result = simulator.run(tickers=tickers, period=args.period)
        
        if args.json:
            output = {
                'strategy': 'active_trading',
                'initial_capital': float(result.initial_capital),
                'final_value': float(result.final_value),
                'total_return': float(result.total_return),
                'annualized_return': float(result.annualized_return),
                'sharpe_ratio': float(result.sharpe_ratio),
                'max_drawdown': float(result.max_drawdown),
                'win_rate': float(result.win_rate),
                'profit_factor': float(result.profit_factor),
                'benchmark_return': float(result.benchmark_return),
                'alpha': float(result.alpha),
                'total_trades': int(result.total_trades),
                'beat_benchmark': bool(result.total_return > result.benchmark_return)
            }
            print(json.dumps(output, indent=2))
        else:
            console.print()
            console.print(simulator.format_results(result))
            console.print(simulator.get_trade_log(result))
            
            if result.total_return > result.benchmark_return:
                console.print(Panel(
                    f"[bold green]✅ STRATEGY BEAT S&P 500![/]\n\n"
                    f"Strategy Return: [green]{result.total_return:+.2%}[/]\n"
                    f"S&P 500 Return:  [yellow]{result.benchmark_return:+.2%}[/]\n"
                    f"Alpha:           [green]{result.alpha:+.2%}[/]",
                    title="🏆 SIMULATION RESULT",
                    border_style="green"
                ))
            else:
                console.print(Panel(
                    f"[bold red]❌ STRATEGY UNDERPERFORMED S&P 500[/]\n\n"
                    f"Strategy Return: [red]{result.total_return:+.2%}[/]\n"
                    f"S&P 500 Return:  [green]{result.benchmark_return:+.2%}[/]\n"
                    f"Alpha:           [red]{result.alpha:+.2%}[/]",
                    title="📉 SIMULATION RESULT",
                    border_style="red"
                ))
    
    return 0


def cmd_longrun(args):
    """Run live trading simulation (scheduled mode)."""
    from src.live_trader import LiveTrader
    
    trader = LiveTrader()
    
    # Initialize mode
    if args.init:
        if not args.json:
            console.print(f"[bold cyan]🚀 Initializing Long Run Simulation[/]")
            console.print(f"Initial Capital: ${args.init:,.2f}")
        
        state = trader.initialize(args.init)
        
        if args.json:
            print(json.dumps({
                'action': 'initialized',
                'initial_capital': state.initial_capital,
                'timestamp': state.initialized_at
            }, indent=2))
        else:
            console.print(f"[green]✅ Initialized! State saved to {trader.state_file}[/]")
        return 0
    
    # Status mode
    if args.status:
        if not trader.load_state():
            console.print("[red]No state found. Initialize first with --init[/]")
            return 1
        
        # Refresh prices for real-time accuracy
        if not args.json:
            console.print("[dim]Fetching latest prices...[/]")
        trader.refresh_prices()
        
        if args.json:
            print(json.dumps(trader.get_status(), indent=2, default=str))
        else:
            console.print(trader.format_status())
        return 0
    
    # Run cycle mode (default)
    if not trader.load_state():
        console.print("[red]No state found. Initialize first with --init[/]")
        return 1
    
    if not args.json:
        console.print(f"[bold cyan]🔄 Running Trading Cycle[/]")
        console.print(f"Execute trades: {'Yes' if args.execute else 'No (dry run)'}")
        console.print()
    
    result = trader.run_cycle(execute=args.execute)
    
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        # Display portfolio summary
        p = result['portfolio']
        console.print(Panel(
            f"Total Value: [bold]${p['total_value']:,.2f}[/]\n"
            f"Cash: ${p['cash']:,.2f} | Positions: ${p['positions_value']:,.2f}\n"
            f"Return: [{'green' if p['total_return'] >= 0 else 'red'}]{p['total_return']:+.2%}[/]",
            title="💰 Portfolio"
        ))
        
        # Sell recommendations
        sells = result['recommendations']['sell']
        if sells:
            console.print(f"\n[bold red]📉 SELL Recommendations ({len(sells)})[/]")
            for rec in sells:
                console.print(f"  • {rec['ticker']}: {rec['reason']}")
        
        # Buy recommendations
        buys = result['recommendations']['buy']
        if buys:
            console.print(f"\n[bold green]📈 BUY Recommendations ({len(buys)})[/]")
            for rec in buys:
                conf = rec['confidence']
                console.print(f"  • {rec['ticker']} ({conf:.0%} conf): {rec['target_shares']} shares @ ${rec['current_price']:.2f}")
                console.print(f"    {rec['reason']}")
        
        if not sells and not buys:
            console.print("\n[dim]No trade recommendations at this time.[/]")
        
        # Executed trades
        if result['executed_trades']:
            console.print(f"\n[bold]✅ Executed Trades ({len(result['executed_trades'])})[/]")
            for trade in result['executed_trades']:
                action_color = 'green' if trade['action'] == 'BUY' else 'red'
                console.print(f"  [{action_color}]{trade['action']}[/] {trade['shares']} {trade['ticker']} @ ${trade['price']:.2f}")
    
    # Send email notification
    if args.email:
        from src.email_notifier import send_trading_summary
        if not args.json:
            console.print("\n[dim]Sending email summary...[/]")
        if send_trading_summary(result, executed=args.execute):
            if not args.json:
                console.print("[green]✅ Email sent![/]")
        else:
            if not args.json:
                console.print("[yellow]⚠️ Email not sent (check SMTP config)[/]")
    
    return 0


def cmd_api(args):
    """Run in API mode (for TUI)."""
    from src.api import StockAPI
    
    api = StockAPI()
    api.run_interactive()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="📊 Stock Market Analysis & Prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze AAPL              Analyze Apple stock
  %(prog)s scan --quick              Quick scan of top 20 stocks
  %(prog)s scan --limit 50           Scan top 50 stocks
  %(prog)s backtest TSLA --period 2y Backtest Tesla strategy
  %(prog)s api                       Run in API mode for TUI
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single ticker')
    analyze_parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    analyze_parser.add_argument('--period', default='1y', help='Data period (default: 1y)')
    analyze_parser.add_argument('--json', action='store_true', help='Output as JSON')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan multiple tickers')
    scan_parser.add_argument('--quick', action='store_true', help='Quick scan (top 20)')
    scan_parser.add_argument('--limit', type=int, help='Limit number of tickers')
    scan_parser.add_argument('--period', default='1y', help='Data period (default: 1y)')
    scan_parser.add_argument('--json', action='store_true', help='Output as JSON')
    scan_parser.set_defaults(func=cmd_scan)
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    backtest_parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    backtest_parser.add_argument('--period', default='2y', help='Data period (default: 2y)')
    backtest_parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    backtest_parser.add_argument('--json', action='store_true', help='Output as JSON')
    backtest_parser.set_defaults(func=cmd_backtest)
    
    # Simulate command
    simulate_parser = subparsers.add_parser('simulate', help='Run trading simulation over historical data')
    simulate_parser.add_argument('--capital', type=float, default=100000, help='Initial capital (default: 100000)')
    simulate_parser.add_argument('--period', default='2y', help='Data period (default: 2y)')
    simulate_parser.add_argument('--max-positions', type=int, default=8, help='Max simultaneous positions (default: 8)')
    simulate_parser.add_argument('--position-size', type=float, default=0.12, help='Position size as fraction (default: 0.12)')
    simulate_parser.add_argument('--tickers', type=int, default=30, help='Number of tickers to trade (default: 30)')
    simulate_parser.add_argument('--enhanced', action='store_true', default=True, help='Use enhanced simulator (default)')
    simulate_parser.add_argument('--json', action='store_true', help='Output as JSON')
    simulate_parser.set_defaults(func=cmd_simulate)
    
    # Long Run command
    longrun_parser = subparsers.add_parser('longrun', help='Run live trading simulation (scheduled mode)')
    longrun_parser.add_argument('--init', type=float, help='Initialize with capital amount')
    longrun_parser.add_argument('--status', action='store_true', help='Show current portfolio status')
    longrun_parser.add_argument('--execute', action='store_true', help='Execute recommended trades')
    longrun_parser.add_argument('--email', action='store_true', help='Send email summary')
    longrun_parser.add_argument('--json', action='store_true', help='Output as JSON')
    longrun_parser.set_defaults(func=cmd_longrun)
    
    # API command
    api_parser = subparsers.add_parser('api', help='Run in API mode')
    api_parser.set_defaults(func=cmd_api)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
