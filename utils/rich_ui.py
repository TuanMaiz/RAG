"""Rich UI components for the RAG chat interface."""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

# Console instance for rich output
console = Console()


def print_user_message(query: str) -> None:
    """Print user query in a styled panel.

    Args:
        query: User's input question
    """
    if not query:
        return

    text = Text(query, style="cyan")
    panel = Panel(
        text,
        title="You",
        title_align="left",
        border_style="cyan",
        padding=(0, 1),
    )
    console.print(panel)


def print_assistant_message(response: str) -> None:
    """Print assistant response in a styled panel with markdown rendering.

    Args:
        response: Assistant's response text (may include markdown)
    """
    if not response:
        return

    # Render as markdown for rich formatting (bold, code, lists, etc.)
    markdown = Markdown(response)
    panel = Panel(
        markdown,
        title="Assistant",
        title_align="left",
        border_style="green",
        padding=(0, 1),
        expand=True,
    )
    console.print(panel)


def print_welcome_message() -> None:
    """Print welcome message when entering chat mode."""
    from rich import box

    panel = Panel(
        "[bold green]Chat mode[/bold green]\n"
        "Type your questions below.\n"
        "Type [bold]quit[/bold], [bold]exit[/bold], or [bold]q[/bold] to exit.",
        box.ROUNDED,
        border_style="blue",
        padding=(1, 2),
    )
    console.print(panel)


def print_goodbye_message() -> None:
    """Print goodbye message when exiting chat mode."""
    console.print("[blue]Goodbye![/blue]")


def get_user_input() -> str:
    """Get user input with rich styling.

    Returns:
        User's input string, stripped of whitespace
    """
    return console.input("[cyan]You: [/]").strip()


def print_error(message: str) -> None:
    """Print error message in red.

    Args:
        message: Error message to display
    """
    console.print(f"[red]{message}[/red]")


def print_info(message: str) -> None:
    """Print info message in blue.

    Args:
        message: Info message to display
    """
    console.print(f"[blue]{message}[/blue]")


def print_success(message: str) -> None:
    """Print success message in green.

    Args:
        message: Success message to display
    """
    console.print(f"[green]{message}[/green]")
