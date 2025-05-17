import flet as ft

COMMON_INPUT_STYLE = {
    "label_style": ft.TextStyle(size=12),
    "text_size": 12,
    "text_style": ft.TextStyle(size=12),

    "content_padding": ft.padding.symmetric(vertical=10, horizontal=10),
    "dense": True,
    "border_color": ft.Colors.BLUE_GREY_200,
    "focused_border_color": ft.Colors.BLUE_GREY_700,
    "fill_color": ft.Colors.GREY_900,
    "filled": True,
}

def create_textfield(label, value, hint_text=None, multiline=False, min_lines=1, max_lines=1, expand=None, col=None, on_change=None, tooltip=None):
    """Helper function to create a TextField with common styling, accepting expand and col."""
    return ft.TextField(
        label=label,
        value=str(value) if value is not None else "",
        hint_text=hint_text,
        multiline=multiline,
        min_lines=min_lines,
        max_lines=max_lines,
        expand=expand,
        col=col,
        on_change=on_change,
        tooltip=tooltip,
        **COMMON_INPUT_STYLE
    )

def create_dropdown(label, value, options, hint_text=None, expand=None, col=None):
    """Helper function to create a Dropdown with common styling, accepting expand. If options contains the special key '__include_none__', a 'None' option will be added at the top."""
    include_none = False
    if isinstance(options, dict) and "__include_none__" in options:
        include_none = True
        options = {k: v for k, v in options.items() if k != "__include_none__"}
    dropdown_options = []
    if include_none:
        dropdown_options.append(ft.dropdown.Option("", text="None"))
    dropdown_options += [ft.dropdown.Option(str(opt_value), text=opt_text) for opt_value, opt_text in options.items()]
    initial_value_key = str(value) if value is not None else "" # Ensure value is string for lookup
    initial_option = next((opt.key for opt in dropdown_options if opt.key == initial_value_key), None)

    return ft.Dropdown(
        label=label,
        value=initial_option, # Set the initial value using the key
        hint_text=hint_text,
        options=dropdown_options,
        expand=expand,
        col=col,
        scale=0.9,
        **COMMON_INPUT_STYLE
    )

def add_section_title(title):
    """Helper function to create a section title and divider."""
    return [
        ft.Container(
            content=ft.Text(title, weight=ft.FontWeight.BOLD, size=16),
            margin=ft.margin.only(bottom=-5)
        ),
        ft.Divider(height=5, thickness=1) # Using height 5 for consistency with sampling page
    ]

# --- Popup Styles ---
POPUP_ACTIONS_ALIGNMENT = ft.MainAxisAlignment.END
POPUP_BORDER_RADIUS = ft.border_radius.all(10) # Example radius, adjust as needed

# Video Player Dialog Dimensions
VIDEO_PLAYER_DIALOG_WIDTH = 640
VIDEO_PLAYER_DIALOG_HEIGHT = 480

# --- Button Styles ---
BTN_STYLE = ft.ButtonStyle(
    text_style=ft.TextStyle(size=14),
    shape=ft.RoundedRectangleBorder(radius=0)
)

BTN_STYLE2 = ft.ButtonStyle(
    text_style=ft.TextStyle(size=12),
    shape=ft.RoundedRectangleBorder(radius=3)
)

BTN_STYLE2_RED = ft.ButtonStyle(
    text_style=ft.TextStyle(size=12),
    shape=ft.RoundedRectangleBorder(radius=3),
    bgcolor=ft.Colors.RED_ACCENT_700,
    color=ft.Colors.WHITE
)

BUTTON_WIDTH = 200
BUTTON_HEIGHT = 40

def create_styled_button(text, on_click=None, col=None, width=BUTTON_WIDTH, height=BUTTON_HEIGHT, button_style=BTN_STYLE, **kwargs):
    """Helper to create a consistently styled button."""
    return ft.ElevatedButton(
        text,
        on_click=on_click,
        style=button_style,
        width=width,
        height=height,
        col=col,
        **kwargs
    ) 