import random


# Function to split text into four parts and trim
def split_and_trim_text(text, part_lenght):
    words = text.split()
    total_word_count = len(words)

    # If the text is shorter than part_lenght, return orignal text
    if total_word_count < part_lenght:
        return [text]

    # Calculate the number of potential part_lenght parts
    num_parts = total_word_count // part_lenght

    trimmed_parts = []
    for i in range(num_parts):
        start_index = i * part_lenght
        end_index = start_index + part_lenght

        # Ensure we do not create a part less than 300 words
        if end_index > total_word_count:
            break

        part = words[start_index:end_index]
        trimmed_parts.append(' '.join(part))
    return trimmed_parts


def create_combined_context(original_row, similar_rows):
    # Split and trim the original text, and create links for each part
    original_parts = split_and_trim_text(original_row['text'], 100)
    original_celex_id = original_row['celex_id']
    original_linked_parts = [f"{part} (Source: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex:{original_celex_id})"
                             for part in original_parts]

    # List to store parts from similar rows
    similar_combined_parts = []
    celex_id_list = []
    # Process each similar row and add its parts to the list
    for similar_row in similar_rows:
        similar_parts = split_and_trim_text(similar_row['text'], 100)
        similar_celex_id = similar_row['celex_id']
        celex_id_list.append(similar_celex_id)
        similar_linked_parts = [f"{part} (Source: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex:{similar_celex_id})"
                                for part in similar_parts]
        similar_combined_parts.extend(similar_linked_parts)

    # Determine the number of parts to take from the original
    if len(original_linked_parts) > 2:
        num_parts_from_original = random.choice([2, 3])
    else:
        num_parts_from_original = len(original_linked_parts)

    # Randomly select parts from the original
    selected_original_parts = random.sample(original_linked_parts, num_parts_from_original)

    # Fill the remaining slots with parts from the similar list
    num_remaining_parts = 4 - num_parts_from_original

    # Ensure not to sample more than available parts
    num_remaining_parts = min(num_remaining_parts, len(similar_combined_parts))
    
    selected_similar_parts = random.sample(similar_combined_parts, num_remaining_parts)

    # Combine and shuffle the selected parts
    combined_parts_with_links = selected_original_parts + selected_similar_parts
    random.shuffle(combined_parts_with_links)

    return combined_parts_with_links, celex_id_list


# Function to calculate the overlap between two lists
def calculate_overlap(list1, list2):
    return len(set(list1).intersection(set(list2)))


# Function to find the top two documents with the most overlap for each document
def find_top_two_overlaps(df):
    top_two_overlaps = []

    for i, concepts_i in enumerate(df['eurovoc_concepts']):
        overlaps = []
        for j, concepts_j in enumerate(df['eurovoc_concepts']):
            if i != j:
                overlap_count = calculate_overlap(concepts_i, concepts_j)
                overlaps.append((j, overlap_count))

        # Sort by overlap count and get the indices of the top two
        sorted_overlaps = sorted(overlaps, key=lambda x: x[1], reverse=True)
        top_two_indices = [idx for idx, count in sorted_overlaps[:2]]
        top_two_overlaps.append(top_two_indices)

    return top_two_overlaps


# Function to process the dataset in chunks
def process_dataset_in_chunks(df, chunk_size):
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)
    chunked_examples = []

    for chunk_idx in range(num_chunks):
        print(f"Chunk {chunk_idx} done")
        # Extract a chunk of the dataset
        chunk = df.iloc[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]

        # Find the top two overlaps for each document in the chunk
        top_two_overlaps = find_top_two_overlaps(chunk)
        # Create combined contexts for each document in the chunk
        combined_contexts = []
        final_celex_list = []
        for i, _ in enumerate(chunk.iterrows()):
            original_row = chunk.iloc[i]
            similar_rows = [chunk.iloc[idx] for idx in top_two_overlaps[i]]
            combined_context, celex_list = create_combined_context(original_row, similar_rows)
            combined_contexts.append(combined_context)
            final_celex_list.append(celex_list)

        chunked_examples.extend(combined_contexts)

    return chunked_examples, final_celex_list
