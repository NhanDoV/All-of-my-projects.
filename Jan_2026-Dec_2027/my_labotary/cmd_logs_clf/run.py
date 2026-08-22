import re, os
import pandas as pd

# read file
with open("logs.txt", "r", encoding="utf-8") as f:
    text = f.read()

# split each event block
event_blocks = re.split(r'\n(?=Event\[\d+\])', text.strip())

rows = []

for block in event_blocks:
    # get event index
    idx_match = re.search(r'Event\[(\d+)\]', block)
    event_idx = int(idx_match.group(1)) if idx_match else None

    row = {"EventIndex": event_idx}
    lines = block.splitlines()

    current_key = None
    description_lines = []

    for line in lines:

        # match "Key: Value"
        m = re.match(r'^\s{2}([^:]+):\s*(.*)$', line)

        if m:
            key = m.group(1).strip()
            value = m.group(2).strip()

            current_key = key

            # special handling for multiline Description
            if key == "Description":
                description_lines = []
                if value:
                    description_lines.append(value)
            else:
                row[key] = value

        else:
            # append multiline description
            if current_key == "Description":
                description_lines.append(line.strip())

    # finalize description
    if description_lines:
        row["Description"] = "\n".join(description_lines).strip()

    rows.append(row)

# convert to dataframe
df = pd.DataFrame(rows)

# set event index
df = df.set_index("EventIndex")
print(df.head())

# filter
df.loc[df['Event ID'] == '41']