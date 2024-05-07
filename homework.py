
import json

def read_data(file_path):
    sentences = []
    current_sentence = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split('\t')
                current_sentence.append((parts[1], parts[2]))
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []

    return sentences

def read_test_data(file_path):
    sentences = []
    current_sentence = ""
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    current_sentence += parts[1] + " "
                else:
                    print(f"Skipping invalid line: {line}")
            else:
                if current_sentence:
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
        if current_sentence:
            sentences.append(current_sentence.strip())
    return sentences


train_file_path = 'data/train'
dev_file_path = 'data/dev'
test_file_path = 'data/test'

# Fetch data from files
train_data = read_data(train_file_path)
dev_data = read_data(dev_file_path)
test_data_raw_sentences = read_test_data(test_file_path)

from collections import Counter

def create_vocabulary(training_data, threshold):
    word_counts = Counter(word for sentence in training_data for word, _ in sentence)
    vocabulary = {'<unk>': 0}
    
    # Add words to vocabulary based on the threshold
    unk_count = 0
    for word, count in word_counts.items():
        if count >= threshold:
            vocabulary[word] = count
        else:
            unk_count += count
    
    # Update the count for <unk>
    vocabulary['<unk>'] += unk_count
    
    return vocabulary


threshold_for_unk = 3

# Create vocabulary
vocab = create_vocabulary(train_data, threshold_for_unk)

def output_vocabulary_to_file(vocabulary, output_file_path='vocab.txt'):
    with open(output_file_path, 'w') as file:
        file.write(f'<unk>\t0\t{vocabulary["<unk>"]}\n')  # Special token <unk>
        index=1
        for word, word_count in sorted(vocabulary.items(), key=lambda x: x[1],reverse=True):
            if word != '<unk>':
                file.write(f"{word}\t{index}\t{word_count}\n")
                index+=1


output_vocab_file_path = 'vocab.txt'

# Output vocabulary to file
output_vocabulary_to_file(vocab, output_vocab_file_path)

# Calculate total size of vocabulary and occurrences of <unk>
total_vocabulary_size = len(vocab)
total_occurrences_of_unk = vocab['<unk>']

# Display results
print(f"Selected threshold for unknown words replacement: {threshold_for_unk}")
print(f"Total size of vocabulary: {total_vocabulary_size}")
print(f"Total occurrences of '<unk>': {total_occurrences_of_unk}")

def replace_infrequent_words(training_data, vocabulary):
    for sentence in training_data:
        for i, (word, tag) in enumerate(sentence):
            if word not in vocabulary or vocabulary[word] == 0:
                sentence[i] = ('<unk>', tag)


replace_infrequent_words(train_data, vocab)

def extract_tags_and_vocabulary(sentences):
    unique_tags = set()
    unique_words = set()

    for sentence in sentences:
        for word, tag in sentence:
            unique_tags.add(tag)
            unique_words.add(word)

    return list(unique_tags), list(unique_words)

unique_tags,unique_words =extract_tags_and_vocabulary(train_data)

def calculate_probabilities_with_smoothing(sentences, tags, vocabulary):
    transition_counts = {}
    emission_counts = {}
    state_counts = {"<START>": len(sentences)}  # Count each sentence start

    # Initialize transition counts for all possible transitions
    for prev_tag in tags + ["<START>"]:
        for tag in tags:
            transition_counts[(prev_tag, tag)] = 0

    # Initialize emission counts for all possible emissions
    for tag in tags:
        for word in vocabulary:
            emission_counts[(tag, word)] = 0

    for sentence in sentences:
        prev_tag = "<START>"  # Start each sentence with the <START> symbol
        for word, tag in sentence:
            # Transition counts
            transition_counts[(prev_tag, tag)] += 1

            # Emission counts
            emission_counts[(tag, word)] += 1

            # State counts
            state_counts[tag] = state_counts.get(tag, 0) + 1

            prev_tag = tag

    # Apply Laplace smoothing to transition and emission counts
    for key in transition_counts.keys():
        transition_counts[key] += 1  # Add-one smoothing

    for key in emission_counts.keys():
        emission_counts[key] += 1  # Add-one smoothing

    # Update state counts for smoothed transitions and emissions
    for tag in tags:
        state_counts[tag] += len(tags)  # For transitions
        state_counts[tag] += len(vocabulary)  # For emissions

    # Calculate probabilities with smoothing
    transition_probs = {k: v / state_counts[k[0]] for k, v in transition_counts.items()}
    emission_probs = {k: v / state_counts[k[0]] for k, v in emission_counts.items()}

    return transition_probs, emission_probs


def save_model(transition_probs, emission_probs, filename):
    # Convert to 2D dictionary structure
    transition_probs_2d = {}
    for (s, s_prime), value in transition_probs.items():
        if s not in transition_probs_2d:
            transition_probs_2d[s] = {}
        transition_probs_2d[s][s_prime] = value

    emission_probs_2d = {}
    for (s, x), value in emission_probs.items():
        if s not in emission_probs_2d:
            emission_probs_2d[s] = {}
        emission_probs_2d[s][x] = value

    model = {
        "transition": transition_probs_2d,
        "emission": emission_probs_2d
    }

    with open(filename, 'w') as file:
        json.dump(model, file, indent=4)


transition_probs, emission_probs= calculate_probabilities_with_smoothing(train_data, unique_tags, unique_words)

save_model(transition_probs, emission_probs, 'hmm.json')

def sentences_to_word_lists(sentences):
    word_lists = []
    for sentence in sentences:
        words = sentence.split()
        word_lists.append(words)
    return word_lists

test_data = sentences_to_word_lists(test_data_raw_sentences)


def greedy_decode(sentence, transition_probs, emission_probs,unique_tags):
    predicted_tags = []
    prev_tag = "<START>"

    for word in sentence:
        max_prob = 0
        best_tag = None

        for tag in unique_tags:
            emission_prob = emission_probs.get((tag, word), 0)
            transition_prob = transition_probs.get((prev_tag, tag), 0)
            prob = emission_prob * transition_prob

            if prob > max_prob:
                max_prob = prob
                best_tag = tag

        if best_tag is None:
            for tag in unique_tags:
                emission_prob = emission_probs.get((tag,'<unk>'), 0)
                transition_prob = transition_probs.get((prev_tag, tag), 0)
                prob = emission_prob * transition_prob

                if prob > max_prob:
                    max_prob = prob
                    best_tag = tag
        predicted_tags.append(best_tag)
        prev_tag = best_tag

    return predicted_tags


def write_predictions_to_file(test_data, transition_probs, emission_probs,unique_tags ,output_file):
    with open(output_file, 'w') as file:
        for sentence in test_data:
            predicted_tags = greedy_decode(sentence, transition_probs, emission_probs,unique_tags)
            
            for index, (word, tag) in enumerate(zip(sentence, predicted_tags), start=1):
                file.write(f"{index}\t{word}\t{tag}\n")
            file.write("\n")  


write_predictions_to_file(test_data, transition_probs, emission_probs,unique_tags, "greedy.out")




def viterbi_decode(sentence, transition_probs, emission_probs, unique_tags):

    V = [{}]
    path = {}


    for tag in unique_tags:
        
        word_prob = emission_probs.get((tag, sentence[0]), emission_probs[(tag, '<unk>')])
        V[0][tag] = transition_probs.get(('<START>', tag), 0) * word_prob
        
        path[tag] = [tag]

   
    for t in range(1, len(sentence)):
        V.append({})
        new_path = {}

        for curr_tag in unique_tags:
            max_prob, best_prev_tag = max(
                (V[t-1][prev_tag] * transition_probs.get((prev_tag, curr_tag), 0) * 
                 emission_probs.get((curr_tag, sentence[t]), emission_probs.get((curr_tag, '<unk>'), 0)), prev_tag)
                for prev_tag in V[t-1]
            )

            V[t][curr_tag] = max_prob
            
            new_path[curr_tag] = path[best_prev_tag] + [curr_tag]

        path = new_path


    max_prob, best_last_tag = max((V[len(sentence) - 1][last_tag], last_tag) for last_tag in V[len(sentence) - 1])


    best_path = path[best_last_tag]

    return best_path


def write_predictions_to_file_viterbi(test_data, transition_probs, emission_probs,unique_tags,output_file):
    with open(output_file, 'w') as file:
        for sentence in test_data:
            predicted_tags = viterbi_decode(sentence, transition_probs, emission_probs,unique_tags)
            
            for index, (word, tag) in enumerate(zip(sentence, predicted_tags), start=1):
                file.write(f"{index}\t{word}\t{tag}\n")
            file.write("\n")  


write_predictions_to_file_viterbi(test_data, transition_probs, emission_probs,unique_tags,"viterbi.out")