from pyboxen import boxen


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKRED = "\033[91m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


COLOR_MAPPING = {
    "blue": Colors.OKBLUE,
    "green": Colors.OKGREEN,
    "cyan": Colors.OKCYAN,
    "red": Colors.OKRED,
}


def print_msg(msg: str, title: str = None, color: str = "blue", box=False):
    if box:
        print(boxen(msg, title=title, color=color, fullwidth=True, style="horizontals"))
    else:
        if title:
            print(
                "\n"
                + "=" * 20
                + f"{title}"
                + "=" * 20
                + "\n"
                + COLOR_MAPPING.get(color, Colors.OKBLUE)
                + msg
                + Colors.ENDC
                + "\n"
                + "=" * 50
                + "\n"
            )
        else:
            print(COLOR_MAPPING.get(color, Colors.OKBLUE) + msg + Colors.ENDC)


def string_to_bool(s):
    return s.lower() in ["true", "1", "t", "y", "yes"]
