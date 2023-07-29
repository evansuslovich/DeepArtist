import json

def text_to_json(input_file, output_file):
    # Read the text file and process the data into a dictionary
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    # arr = [[] * len(lines[0].split(","))] * len(lines)
    arr = []
    for line in lines:
        temp = line.split(",")
        arr.append(temp)

    json_data = json.dumps(arr, indent=4)

    # Write the JSON data to a new file
    with open(output_file, 'w') as f:
        f.write(json_data)

if __name__ == "__main__":
    input_file = "MetObjects.txt"
    output_file = "output.json"

    text_to_json(input_file, output_file)
import csv
import json

def json_to_csv(json_file, csv_file):
    # Read JSON data from the file
    with open(json_file, 'r') as json_data:
        data = json.load(json_data)

    # Write CSV data to the file
    with open(csv_file, 'w', newline='') as csv_data:
        # Create a CSV writer
        csv_writer = csv.writer(csv_data)

        # Write each row of data
        for row in data:
            csv_writer.writerow(row)

if __name__ == "__main__":
    json_file_path = "output.json"
    csv_file_path = "data.csv"

    json_to_csv(json_file_path, csv_file_path)
