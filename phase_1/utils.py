def save_final_output(filename, agent_name, **sections):
    """
    Append structured output to a file, grouped by agent name.
    Each keyword argument is a section, e.g., steps="...", response="...", evaluation="..."
    """
    with open(filename, "a") as f:
        f.write(f"\n=== {agent_name} Output ===\n\n")
        for section, content in sections.items():
            f.write(f"--- {section.replace('_', ' ').title()} ---\n")
            f.write(str(content) + "\n\n")