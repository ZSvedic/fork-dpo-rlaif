import torch
import transformers as hftf
import sys
import os
import time
import huggingface_hub

if len(sys.argv) > 1:
    mode = sys.argv[1]
    base_model = sys.argv[2]
    model_name = sys.argv[3]
else:
    mode = "evaluate"
    base_model = model_name = 'meta-llama/Llama-2-7b-chat-hf'

def load_tokenizer_model(model_name):
    print(f"===== Base model: {base_model}")
    print(f"===== Model: {model_name}")

    # Load model in 2 steps: base first, then checkpoint
    model = hftf.AutoModelForCausalLM.from_pretrained(base_model, device_map="cuda:0")
    tokenizer = hftf.AutoTokenizer.from_pretrained(base_model, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if model_name!=base_model:
        checkpoint_path = os.path.join(model_name, "policy.pt")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict['state'])

    print(f'Allocated GPU memory: {torch.cuda.memory_allocated() / (1024*1024):,.1f} MB')

    return tokenizer, model

def calc_model_avg_len(tokenizer, model):

    questions = [
        "How much is 2+3?",
        "What is the color of the sky?",
        "What is the capital of France?",
        "What is the boiling point of water?",
        "Who wrote 'To Kill a Mockingbird'?",
        "What is the largest planet in our solar system?",
        "What is the chemical symbol for gold?",
        "How many continents are there?",
        "What is the speed of light?",
        "Who painted the Mona Lisa?",
        "What is the smallest prime number?",
        "What is the main ingredient in guacamole?",
        "What is the square root of 64?",
        "What is the currency of Japan?",
        "Who discovered penicillin?",
        "What is the tallest mountain in the world?",
        "What is the primary language spoken in Brazil?",
        "What is the freezing point of water?",
        "What is the largest mammal?",
        "What is the capital of Japan?"
    ]

    messages = [ [{"role": "user", "content": q}] for q in questions]

    prompts = [tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            for m in messages]

    inputs = tokenizer(prompts, return_tensors='pt', padding=True).to('cuda:0')
    inputs_tok_len = inputs["input_ids"].shape[1]
    # print(inputs_tok_len)

    # text_streamer = TextStreamer(tokenizer)
    # sequences = model.generate(input_ids = inputs[0], streamer = text_streamer, max_new_tokens = 1024, use_cache = True)
    results = model.generate(**inputs, max_new_tokens = 200, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    sequences = tokenizer.batch_decode(results[:, inputs_tok_len:], skip_special_tokens=True)

    averages = []
    for p, answer in zip(prompts, sequences):
        print(f"PROMPT: {p}\nANSWER: {answer}")
        print('------------------------------------------------------------------------------------')
        averages.append(len(answer))

    total_average = sum(averages)/len(averages)
    print(f"Average {model.name_or_path} answer length for {len(averages)} questions: {total_average:.2f} characters")

    return total_average

# Load the tokenizer and model:
tokenizer, model = load_tokenizer_model(model_name)
model.eval()

# Then evaluate and run the mode:
avg_len = calc_model_avg_len(tokenizer, model)

if mode=="upload":
    # Save model locally
    save_path = "./CACHE/saved_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    # Save to HF
    model_str = base_model.split('/')[1]
    repo_id = f"ZSvedic/dpo-rlaif-{model_str}-len{int(avg_len)}-{time.strftime('%Y-%m-%d-%H-%M')}"
    huggingface_hub.create_repo(repo_id, exist_ok=True)
    huggingface_hub.upload_folder(
        folder_path=save_path,
        path_in_repo=".",
        repo_id=repo_id,
        commit_message="Add fine-tuned model"
    )
elif mode!="evaluate":
    print(f"ERROR: Mode needs to be evaluate or upload, there is no mode {mode}")
