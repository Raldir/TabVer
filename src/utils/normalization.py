import re


def normalize_subclaim_id(entry_id):
    if str(entry_id).endswith("000"):
        # matches = re.findall("0{7}\d+0{7}(?!.*[0-9])", str(entry_id))
        matches = re.findall("0{3}\d0{3}$", str(entry_id))
        if matches:
            entry_id = str(entry_id).rsplit(str(matches[0]))[0]
    return int(entry_id)

def normalize_text(text):
    # Define a dictionary for converting numbers to words
    num_to_word = {
    1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
    7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve',
    13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen', 17: 'seventeen',
    18: 'eighteen', 19: 'nineteen', 20: 'twenty',
    21: 'twenty-one', 22: 'twenty-two', 23: 'twenty-three', 24: 'twenty-four',
    25: 'twenty-five', 26: 'twenty-six', 27: 'twenty-seven', 28: 'twenty-eight',
    29: 'twenty-nine', 30: 'thirty'
}

    # Define a list of SI units and their corresponding words
    si_units = {
        'meters': 'meters', 'm': 'meters',
        'inches': 'inches', 'in': 'inches',
        'feet': 'feet', 'ft': 'feet',
        'liters': 'liters', 'L': 'liters',
        'kilometers': 'kilometers', 'km': 'kilometers',
        'Pct': 'Percent'  # Add the mapping for 'Pct' to 'Percent'
        # Add more units as needed
    }

    # Normalize ordinal numbers from 1st to 20th
    for i in range(1, 21):
        if i <= 3:
            ordinal = f"{i}st"
        elif i <= 13:
            ordinal = f"{i}th"
        else:
            ones_digit = i % 10
            if ones_digit == 1:
                ordinal = f"{i}st"
            elif ones_digit == 2:
                ordinal = f"{i}nd"
            elif ones_digit == 3:
                ordinal = f"{i}rd"
            else:
                ordinal = f"{i}th"
        text = re.sub(r'\b{}\b'.format(ordinal), num_to_word[i], text)


    # Define a pattern to match whole numbers in the text
    num_pattern = r'\b(\d+)\b'
    
    # Normalize numbers to words (1-20)
    matches = re.findall(num_pattern, text)
    for match in matches:
        num = int(match)
        if num in num_to_word:
            text = text.replace(match, num_to_word[num])

    # Normalize SI units to words
    for unit, word in si_units.items():
        text = re.sub(r'(\d+)\s*{}\s*'.format(unit), r'\1 {} '.format(word), text)

    # Add spaces around brackets
    text = re.sub(r'\(', ' ( ', text)
    text = re.sub(r'\)', ' ) ', text)

    return text


if "__main__" == __name__:
    # Example usage:
    input_text = "The 1st, 2nd, and 3rd meters are 5 meters (2 inches). There are 10 apples and 15 pears."
    normalized_text = normalize_text(input_text)
    print(normalized_text)
