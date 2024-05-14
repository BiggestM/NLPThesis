import csv

input_file = 'Poems_Dataset.csv'
output_file_txt = 'poem_contents.txt'

# Open input CSV file for reading
with open(input_file, 'r', newline='', encoding='utf-8') as f_in, \
        open(output_file_txt, 'w', encoding='utf-8') as f_out_txt:
    # Create CSV reader object
    reader = csv.DictReader(f_in)

    # Iterate over each row in the input CSV file
    for row in reader:
        # Extract poem content and write it to the output text file
        poem_content = row['Poem Content']
        f_out_txt.write(poem_content + '\n')

print("Poem contents extracted and saved to:", output_file_txt)
