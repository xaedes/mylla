
from mylla.llama_chat import *

def main():
    chat = LlamaChat(ChatConfig(prompt_fn='chat-with-bob.txt', llama_cfg=LlamaConfig(model='13b')))

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
                    chat.process_input(echo=False)
                is_interacting = False
            else:
                token = chat.sample(ignore=["\\"])
            chat.process_input()


if __name__ == "__main__": main()
