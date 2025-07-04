from pydantic import BaseModel, Field
from textual.design import ColorSystem
import yaml

from elia_chat.locations import theme_directory


class Theme(BaseModel):
    name: str = Field(exclude=True)
    primary: str
    secondary: str | None = None
    background: str | None = None
    surface: str | None = None
    panel: str | None = None
    warning: str | None = None
    error: str | None = None
    success: str | None = None
    accent: str | None = None
    dark: bool = True
    variables: dict

    def to_color_system(self) -> ColorSystem:
        """Convert this theme to a ColorSystem."""
        return ColorSystem(
            **self.model_dump(
                exclude={
                    "text_area",
                    "syntax",
                    "variable",
                    "url",
                    "method",
                }
            )
        )


def load_user_themes() -> dict[str, Theme]:
    """Load user themes from "~/.config/elia/themes".

    Returns:
        A dictionary mapping theme names to theme objects.
    """
    themes: dict[str, Theme] = {}
    for path in theme_directory().iterdir():
        path_suffix = path.suffix
        if path_suffix == ".yaml" or path_suffix == ".yml":
            with path.open() as theme_file:
                theme_content = yaml.load(theme_file, Loader=yaml.FullLoader) or {}
                try:
                    themes[theme_content["name"]] = Theme(**theme_content)
                except KeyError:
                    raise ValueError(
                        f"Invalid theme file {path}. A `name` is required."
                    )
    return themes


BUILTIN_THEMES: dict[str, Theme] = {
    "textual": Theme(
        name="textual",
        primary="#004578",
        secondary="#0178D4",
        warning="#ffa62b",
        error="#ba3c5b",
        success="#4EBF71",
        accent="#ffa62b",
        dark=True,
        variables = {},
    ),
    "monokai": Theme(
        name="monokai",
        primary="#F92672",  # Pink
        secondary="#66D9EF",  # Light Blue
        warning="#FD971F",  # Orange
        error="#F92672",  # Pink (same as primary for consistency)
        success="#A6E22E",  # Green
        accent="#AE81FF",  # Purple
        background="#272822",  # Dark gray-green
        surface="#3E3D32",  # Slightly lighter gray-green
        panel="#3E3D32",  # Same as surface for consistency
        dark=True,
        variables={},
    ),
    "nautilus": Theme(
        name="nautilus",
        primary="#0077BE",  # Ocean Blue
        secondary="#20B2AA",  # Light Sea Green
        warning="#FFD700",  # Gold (like sunlight on water)
        error="#FF6347",  # Tomato (like a warning buoy)
        success="#32CD32",  # Lime Green (like seaweed)
        accent="#FF8C00",  # Dark Orange (like a sunset over water)
        dark=True,
        background="#001F3F",  # Dark Blue (deep ocean)
        surface="#003366",  # Navy Blue (shallower water)
        panel="#005A8C",  # Steel Blue (water surface),
        variables={},
    ),
    "galaxy": Theme(
        name="galaxy",
        primary="#8A2BE2",  # Improved Deep Magenta (Blueviolet)
        secondary="#a684e8",
        warning="#FFD700",  # Gold, more visible than orange
        error="#FF4500",  # OrangeRed, vibrant but less harsh than pure red
        success="#00FA9A",  # Medium Spring Green, kept for vibrancy
        accent="#FF69B4",  # Hot Pink, for a pop of color
        dark=True,
        background="#0F0F1F",  # Very Dark Blue, almost black
        surface="#1E1E3F",  # Dark Blue-Purple
        panel="#2D2B55",  # Slightly Lighter Blue-Purple
        variables = {},
    ),
    "nebula": Theme(
        name="nebula",
        primary="#4169E1",  # Royal Blue, more vibrant than Midnight Blue
        secondary="#9400D3",  # Dark Violet, more vibrant than Indigo Dye
        warning="#FFD700",  # Kept Gold for warnings
        error="#FF1493",  # Deep Pink, more nebula-like than Crimson
        success="#00FF7F",  # Spring Green, slightly more vibrant
        accent="#FF00FF",  # Magenta, for a true neon accent
        dark=True,
        background="#0A0A23",  # Dark Navy, closer to a night sky
        surface="#1C1C3C",  # Dark Blue-Purple
        panel="#2E2E5E",  # Slightly Lighter Blue-Purple
        variables = {},
    ),
    "alpine": Theme(
        name="alpine",
        primary="#4A90E2",  # Clear Sky Blue
        secondary="#81A1C1",  # Misty Blue
        warning="#EBCB8B",  # Soft Sunlight
        error="#BF616A",  # Muted Red
        success="#A3BE8C",  # Alpine Meadow Green
        accent="#5E81AC",  # Mountain Lake Blue
        dark=True,
        background="#2E3440",  # Dark Slate Grey
        surface="#3B4252",  # Darker Blue-Grey
        panel="#434C5E",  # Lighter Blue-Grey
        variables = {},
    ),
    "cobalt": Theme(
        name="cobalt",
        primary="#334D5C",  # Deep Cobalt Blue
        secondary="#4878A6",  # Slate Blue
        warning="#FFAA22",  # Amber, suitable for warnings related to primary
        error="#E63946",  # Red, universally recognized for errors
        success="#4CAF50",  # Green, commonly used for success indication
        accent="#D94E64",  # Candy Apple Red
        dark=True,
        surface="#27343B",  # Dark Lead
        panel="#2D3E46",  # Storm Gray
        background="#1F262A",  # Charcoal
        variables = {},
    ),
    "twilight": Theme(
        name="twilight",
        primary="#367588",
        secondary="#5F9EA0",
        warning="#FFD700",
        error="#FF6347",
        success="#00FA9A",
        accent="#FF7F50",
        dark=True,
        background="#191970",
        surface="#3B3B6D",
        panel="#4C516D",
        variables = {},
    ),
    "hacker": Theme(
        name="hacker",
        primary="#00FF00",  # Bright Green (Lime)
        secondary="#32CD32",  # Lime Green
        warning="#ADFF2F",  # Green Yellow
        error="#FF4500",  # Orange Red (for contrast)
        success="#00FA9A",  # Medium Spring Green
        accent="#39FF14",  # Neon Green
        dark=True,
        background="#0D0D0D",  # Almost Black
        surface="#1A1A1A",  # Very Dark Gray
        panel="#2A2A2A",  # Dark Gray
        variables = {},
    ),
}
