import re
from web_utils import get_concept_id

def extract_json_substring(text):
    match = re.search(r'```json(.*?)```', text, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_domain_text(input_string):
    pattern = r"<(?P<domain>[^>]+)>(?P<text>.*?)</\1>"
    matches = re.finditer(pattern, input_string)

    results = [(match.group("domain"), match.group("text")) for match in matches]
    return results

# NLP functions
def formulate_ner_result(text, gpt_results):
    terms = [{"text": term.strip()} for term in text.split() if term]  # Split text into terms
    negate_cues = []  # Placeholder for negation cues
    return terms, negate_cues


def trans4display(text, terms, concept_set):
    """
    Map terms to their respective concept IDs, assign concept sets if new, and generate display text.
    """
    display = text
    for term in terms:
        term_text = term["text"]
        concept_data = get_concept_id(term_text)
        concept_id = concept_data["concept_id"]
        concept_name = concept_data["concept_name"]

        # Assign the concept ID and name to the term
        term["concept_id"] = concept_id
        term["name"] = concept_name

        # Check if the term needs a new concept set
        if concept_id and f"{term_text} {concept_id}" not in concept_set:
            concept_set_id = len(concept_set) + 1
            concept_set[f"{term_text} {concept_id}"] = concept_set_id
            term["vocabulary_id"] = concept_set_id
        else:
            term["vocabulary_id"] = concept_set.get(f"{term_text} {concept_id}", 0)

        # Generate display text with concept mapping
        if concept_id:  # Valid concept
            display += (
                f"\n<mark data-entity=\"{term.get('domain', 'unknown')}\" "
                f"concept-id=\"{concept_id}\" "
                f"vocabulary-id=\"{term['vocabulary_id']}\">"
                f"{term_text} <b><i>{concept_name}</i></b>"
                f"</mark>"
            )
        else:  # No valid concept
            display += f"\n<mark data-entity=\"{term.get('domain', 'unknown')}\">{term_text}</mark>"
    return display


def translate_by_block_seg_ner_concept_mapping(text, nlp, client, concept_set):
    if not text:
        return []

    spas = []
    paragraphs = text.split('\n')
    for p in paragraphs:
        if not p.strip():
            continue

        pa = {'sents': []}
        block_text = [sent.text for sent in nlp(p).sents]

        for s in block_text:
            gpt_results = client.get_chatgpt_response(s)

            # Remove tags
            s = re.sub(r"<([\s\S]*?)>|</([\s\S]*?)>", " ", s)
            s = re.sub(r" {2,}", " ", s)
            doc = nlp(" " + s + " ")

            # Extract terms and negation cues
            terms, negate_cues = formulate_ner_result(doc.text, gpt_results)

            # Display with dynamic concept mapping
            try:
                display = trans4display(doc.text, terms, concept_set)
            except Exception as ex:
                print(f"Error in trans4display: {ex}")
                display = ""

            sent = {'text': doc.text, 'terms': terms, 'display': display, 'negate_cues': negate_cues}
            pa['sents'].append(sent)

        spas.append(pa)
    return spas