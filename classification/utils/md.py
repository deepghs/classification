import io

import pandas as pd


def markdown_to_df(markdown_text: str) -> pd.DataFrame:
    lines = markdown_text.strip().split('\n')
    header = lines[0]
    data = lines[2:]
    header = header.strip('|').split('|')
    header = [col.strip() for col in header]
    data = [row.strip('|').split('|') for row in data]
    data = [[cell.strip() for cell in row] for row in data]

    csv_string = io.StringIO()
    csv_string.write(','.join(header) + '\n')
    for row in data:
        csv_string.write(','.join(row) + '\n')
    csv_string.seek(0)
    df = pd.read_csv(csv_string)
    return df
