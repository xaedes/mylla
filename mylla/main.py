
from mylla.llama_chat import *

def main():
    chat = ParallelChat(ParallelChatConfig(
        main = ChatConfig(prompt_fn='chat-with-bob.txt', llama_cfg=LlamaConfig(model='7b')),
        others = [
            PartConfig(w=0.5, cfg=ChatConfig(prompt_fn='chat-with-bob2.txt', llama_cfg=LlamaConfig(model='7b')))
        ]
    ))

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
                    tkns = chat.main.llama.tokenize(' ' + txt)
                    for i in range(1, tkns.n):
                        token = tkns.tokens[i]
                        chat.main.insert_token(token)
                        for other in chat.others:
                            other.insert_token(token)
                    chat.main.process_input(echo=False)
                    for other in chat.others:
                        other.process_input(echo=False)
                is_interacting = False
            else:
                for cfg,other in zip(chat.cfg.others, chat.others):
                    chat.main.patch_logits(other, cfg.w, cfg.op)
                token = chat.main.sample(ignore=["\\"])
                for other in chat.others:
                    other.insert_token(token)
            chat.main.process_input()
            for other in chat.others:
                other.process_input(echo=False)


if __name__ == "__main__": main()
