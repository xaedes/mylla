
from mylla.llama_chat import *

def main():
    chat = LlamaChat(ChatConfig(prompt_fn='chat-with-bob.txt', llama_cfg=LlamaConfig(model='13b')))
    chat2 = LlamaChat(ChatConfig(prompt_fn='chat-with-bob2.txt', llama_cfg=LlamaConfig(model='13b')))

    chat2.load_state()
    if chat2.process_input() > 0: chat2.save_state()

    chat.load_state()
    if chat.process_input() > 0: chat.save_state()

    n_gen = 128

    is_interacting = False

    def sigint_handler(signal, frame):
        nonlocal is_interacting
        if is_interacting:
            sys.exit(130)
        is_interacting = True

    signal.signal(signal.SIGINT, sigint_handler)

    while True:
        is_interacting = True
        for k in range(n_gen):
            if is_interacting:
                txt = input()
                if len(txt) > 0:
                    txt += "\n"
                    tkns = chat.llama.tokenize(' ' + txt)
                    for i in range(1, tkns.n):
                        token = tkns.tokens[i]
                        chat.insert_token(token)
                        chat2.insert_token(token)
                    chat.process_input(echo=False)
                    chat2.process_input(echo=False)
                is_interacting = False
            else:
                chat.patch_logits(chat2, 0.5)
                token = chat.sample(ignore=["\\"])
                chat2.insert_token(token)
            chat.process_input()
            chat2.process_input(echo=False)


if __name__ == "__main__": main()
