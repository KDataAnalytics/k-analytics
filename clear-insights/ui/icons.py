def svg_icon(name: str, size: int = 22) -> str:
    icon = ""
    if name == "chart":
        icon = (
            '<svg width="{s}" height="{s}" viewBox="0 0 24 24" aria-hidden="true">'
            '<rect x="3" y="12" width="4" height="9" rx="1" fill="#5fbf93"/>'
            '<rect x="10" y="8" width="4" height="13" rx="1" fill="#5fbf93"/>'
            '<rect x="17" y="4" width="4" height="17" rx="1" fill="#5fbf93"/>'
            "</svg>"
        )
    elif name == "trend_up":
        icon = (
            '<svg width="{s}" height="{s}" viewBox="0 0 24 24" aria-hidden="true">'
            '<path d="M4 16l6-6 4 4 6-8" fill="none" stroke="#6aa6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
            '<circle cx="4" cy="16" r="2" fill="#6aa6ff"/>'
            '<circle cx="10" cy="10" r="2" fill="#6aa6ff"/>'
            '<circle cx="14" cy="14" r="2" fill="#6aa6ff"/>'
            '<circle cx="20" cy="6" r="2" fill="#6aa6ff"/>'
            "</svg>"
        )
    elif name == "search":
        icon = (
            '<svg width="{s}" height="{s}" viewBox="0 0 24 24" aria-hidden="true">'
            '<circle cx="10" cy="10" r="6" fill="#6aa6ff"/>'
            '<rect x="15" y="15" width="7" height="3" rx="1.5" transform="rotate(45 15 15)" fill="#6aa6ff"/>'
            "</svg>"
        )
    elif name == "target":
        icon = (
            '<svg width="{s}" height="{s}" viewBox="0 0 24 24" aria-hidden="true">'
            '<circle cx="12" cy="12" r="9" fill="none" stroke="#f3b23a" stroke-width="2"/>'
            '<circle cx="12" cy="12" r="5" fill="none" stroke="#f3b23a" stroke-width="2"/>'
            '<circle cx="12" cy="12" r="2" fill="#f3b23a"/>'
            "</svg>"
        )
    elif name == "cluster":
        icon = (
            '<svg width="{s}" height="{s}" viewBox="0 0 24 24" aria-hidden="true">'
            '<circle cx="7" cy="8" r="4" fill="#c08adf"/>'
            '<circle cx="16" cy="7" r="3.5" fill="#c08adf"/>'
            '<circle cx="14" cy="16" r="4.5" fill="#c08adf"/>'
            "</svg>"
        )
    elif name == "bulb":
        icon = (
            '<svg width="{s}" height="{s}" viewBox="0 0 24 24" aria-hidden="true">'
            '<path d="M12 3a7 7 0 00-4 12.7V19a2 2 0 002 2h4a2 2 0 002-2v-3.3A7 7 0 0012 3z" fill="#f3b23a"/>'
            '<rect x="9" y="19" width="6" height="2" rx="1" fill="#d99a2b"/>'
            "</svg>"
        )
    elif name == "download":
        icon = (
            '<svg width="{s}" height="{s}" viewBox="0 0 24 24" aria-hidden="true">'
            '<path d="M12 3v10" stroke="#6aa6ff" stroke-width="2" stroke-linecap="round"/>'
            '<path d="M8 9l4 4 4-4" fill="none" stroke="#6aa6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
            '<rect x="4" y="17" width="16" height="4" rx="1.5" fill="#6aa6ff"/>'
            "</svg>"
        )
    elif name == "alert":
        icon = (
            '<svg width="{s}" height="{s}" viewBox="0 0 24 24" aria-hidden="true">'
            '<circle cx="12" cy="12" r="9" fill="#f28b82"/>'
            '<rect x="11" y="6" width="2" height="9" rx="1" fill="#ffffff"/>'
            '<rect x="11" y="16.5" width="2" height="2" rx="1" fill="#ffffff"/>'
            "</svg>"
        )
    else:
        return ""
    return icon.format(s=size)


def heading_html(text: str, icon_name: str, level: int = 2) -> str:
    tag = f"h{level}"
    icon = svg_icon(icon_name, size=22)
    return (
        f"<{tag} style=\"display:flex;align-items:center;gap:10px;margin:0 0 0.5rem 0;\">"
        f"{icon}<span>{text}</span>"
        f"</{tag}>"
    )


def label_html(text: str, icon_name: str) -> str:
    icon = svg_icon(icon_name, size=18)
    return (
        "<span style=\"display:inline-flex;align-items:center;gap:8px;\">"
        f"{icon}<span>{text}</span>"
        "</span>"
    )
