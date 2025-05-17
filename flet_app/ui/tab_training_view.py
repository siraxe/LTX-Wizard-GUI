import flet as ft
from .pages.training_config import get_training_config_page_content
from .pages.training_sampling import get_training_sampling_page_content
from ui.popups import dataset_not_selected
import os
import yaml
import subprocess

# =====================
# Data/Utility Functions
# =====================

def save_training_config_to_yaml(training_tab_container):
    """
    Extracts the config from the UI and saves it as a YAML file.
    Returns the output path and the YAML dictionary.
    """
    from ui.utils.utils_top_menu import TopBarUtils
    yaml_dict = TopBarUtils.build_yaml_config_from_ui(training_tab_container)
    out_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "assets", "config_to_train.yaml")
    )

    class InlineListDumper(yaml.SafeDumper):
        pass

    def repr_inline_list(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    InlineListDumper.add_representer(list, yaml.SafeDumper.represent_list)

    def custom_representer(dumper, data):
        if hasattr(dumper, '_current_key') and dumper._current_key == 'video_dims':
            return repr_inline_list(dumper, data)
        return yaml.SafeDumper.represent_list(dumper, data)

    def represent_mapping(self, tag, mapping, flow_style=None):
        value = []
        for item_key, item_value in mapping.items():
            self._current_key = item_key
            node_key = self.represent_data(item_key)
            node_value = self.represent_data(item_value)
            value.append((node_key, node_value))
        return yaml.MappingNode(tag, value, flow_style=flow_style)

    InlineListDumper.represent_mapping = represent_mapping
    InlineListDumper.add_representer(list, custom_representer)
    InlineListDumper.add_representer(TopBarUtils.QuotedString, TopBarUtils.quoted_presenter)

    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.dump(
            TopBarUtils.quote_all_strings(yaml_dict),
            f,
            sort_keys=False,
            allow_unicode=True,
            Dumper=InlineListDumper
        )
    return out_path, yaml_dict

def run_training_batch_file():
    """
    Runs the training batch file to start the training process.
    Returns the batch file path.
    """
    bat_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "start_training.bat")
    )
    print(f"DEBUG: Batch file path: {bat_path}")
    subprocess.run(bat_path, shell=True)
    return bat_path

def is_dataset_selected(config_page_content):
    """
    Checks if a dataset is selected in the config page content.
    Returns the selected dataset or None.
    """
    dataset_block = getattr(config_page_content, 'dataset_block', None)
    if dataset_block and hasattr(dataset_block, 'get_selected_dataset'):
        return dataset_block.get_selected_dataset()
    return None

# =====================
# GUI-Building Functions
# =====================

def build_navigation_rail(on_nav_change):
    """
    Builds the sub-navigation rail for the training tab.
    """
    config_dest_content = ft.Container(
        content=ft.Text("Config", size=10),
        padding=ft.padding.symmetric(vertical=0, horizontal=0),
        alignment=ft.alignment.center,
        expand=True
    )
    sampling_dest_content = ft.Container(
        content=ft.Text("Sampling", size=10),
        padding=ft.padding.symmetric(vertical=0, horizontal=0),
        alignment=ft.alignment.center,
        expand=True
    )
    return ft.NavigationRail(
        selected_index=0,
        on_change=on_nav_change,
        bgcolor=ft.Colors.TRANSPARENT,
        indicator_color=ft.Colors.TRANSPARENT,
        indicator_shape=ft.RoundedRectangleBorder(radius=0),
        label_type=ft.NavigationRailLabelType.NONE,
        destinations=[
            ft.NavigationRailDestination(icon=config_dest_content),
            ft.NavigationRailDestination(icon=sampling_dest_content),
        ]
    )

def build_main_content_row(sub_navigation_rail, content_area):
    """
    Builds the main content row containing the navigation rail and content area.
    """
    return ft.Row(
        [
            sub_navigation_rail,
            ft.VerticalDivider(),
            content_area,
        ],
        expand=True,
        spacing=0,
        vertical_alignment=ft.CrossAxisAlignment.START
    )

def build_bottom_app_bar(on_start_click):
    """
    Builds the bottom app bar with the Start button.
    """
    return ft.BottomAppBar(
        bgcolor=ft.Colors.BLUE_GREY_900,
        height=60,
        content=ft.Row(
            [
                ft.Container(
                    ft.ElevatedButton(
                        "Start",
                        on_click=on_start_click,
                        style=ft.ButtonStyle(
                            text_style=ft.TextStyle(size=14),
                            shape=ft.RoundedRectangleBorder(radius=0)
                        ),
                        width=150,
                        height=40,
                    ),
                    alignment=ft.alignment.center_right,
                    expand=True,
                ),
            ],
            expand=True,
        ),
    )

def build_main_container(main_content_row, bottom_app_bar):
    """
    Wraps the main content and bottom bar in a Stack for sticky footer.
    """
    return ft.Container(
        padding=ft.padding.only(top=0, bottom=0),
        content=ft.Stack(
            [
                ft.Container(main_content_row, expand=True),
                ft.Container(
                    bottom_app_bar,
                    left=0,
                    right=0,
                    bottom=0,
                    alignment=ft.alignment.bottom_center
                ),
            ],
            expand=True,
        ),
        expand=True
    )

# =====================
# Main Entry Point
# =====================

def get_training_tab_content(page: ft.Page):
    """
    Entry point for building the Training tab content. Sets up navigation, content, and event handlers.
    """
    page.snack_bar = ft.SnackBar(content=ft.Text("Training tab loaded! (debug)"), open=True)
    page.update()

    # Initialize config and sampling page content
    config_page_content = get_training_config_page_content()
    sampling_page_content = get_training_sampling_page_content()

    # Content area container
    content_area = ft.Container(
        content=config_page_content,
        expand=True,
        alignment=ft.alignment.top_left,
        padding=ft.padding.all(0)
    )

    def on_nav_change(e):
        selected_idx = e.control.selected_index
        if selected_idx == 0:
            content_area.content = config_page_content
        elif selected_idx == 1:
            content_area.content = sampling_page_content
        content_area.update()

    sub_navigation_rail = build_navigation_rail(on_nav_change)
    main_content_row = build_main_content_row(sub_navigation_rail, content_area)

    def handle_training_output(page=None):
        """
        Handles saving config and running training, with error handling and user feedback.
        """
        try:
            training_tab_container = None
            if page is not None:
                training_tab_container = getattr(page, 'training_tab_container', None)
            if training_tab_container is None:
                msg = "Error: training_tab_container not found."
                if page is not None:
                    page.snack_bar = ft.SnackBar(content=ft.Text(msg), open=True)
                    page.update()
                return
            out_path, _ = save_training_config_to_yaml(training_tab_container)
            bat_path = run_training_batch_file()
            msg = f"Saved config to {out_path}\nBatch file: {bat_path}\nTraining started."
            if page is not None:
                page.snack_bar = ft.SnackBar(content=ft.Text(msg), open=True)
                page.update()
        except Exception as e:
            msg = f"Error: {e}"
            if page is not None:
                page.snack_bar = ft.SnackBar(content=ft.Text(msg), open=True)
                page.update()

    def handle_start_click(e):
        """
        Handles the Start button click: checks dataset selection and triggers training.
        """
        dataset_selected = is_dataset_selected(config_page_content)
        if not dataset_selected:
            def on_confirm():
                handle_training_output(e.page)
            dataset_not_selected.show_dataset_not_selected_dialog(
                e.page, "Dataset not selected, proceed?", on_confirm
            )
        else:
            handle_training_output(e.page)

    bottom_app_bar = build_bottom_app_bar(handle_start_click)
    main_container = build_main_container(main_content_row, bottom_app_bar)

    # Attach references for later extraction
    main_container.config_page_content = config_page_content
    main_container.sampling_page_content = sampling_page_content
    if hasattr(config_page_content, 'dataset_block'):
        main_container.dataset_page_content = config_page_content.dataset_block
    page.training_tab_container = main_container
    return main_container 