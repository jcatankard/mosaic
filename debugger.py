from streamlit.web import bootstrap
import pathlib


def debug(app_entry_file: str = "app.py") -> None:
    project_root = pathlib.Path().parent.resolve()
    path = f"{project_root}/{app_entry_file}"
    bootstrap.run(path, True, [], {})


if __name__ == "__main__":
    debug()
