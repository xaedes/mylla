
from mylla.llama_chat import *

def main():
    ops = {'min': min, 'max': max}
    chat = ParallelChat(ParallelChatConfig(
        chats = [
            ChatConfig(prompt_fn='chat-with-bob.txt', llama_cfg=LlamaConfig(model='7b')),
            ChatConfig(prompt_fn='chat-with-bob2.txt', llama_cfg=LlamaConfig(model='7b')),
        ],
        patch = [
            PatchConfig(w=0.5, op='min', a=0, b=1),
            PatchConfig(w=1/3., op='max', a=0, b=1)
        ]
    ), ops)
    print(json.dumps(asdict(chat.cfg), indent=4))

    chat.init_from_saved()

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
                    chat.insert_text(txt)
                    chat.process_input(echo_main=False)
                is_interacting = False
            else:
                chat.apply_patch()
                chat.sample()
                chat.process_input()


if __name__ == "__main__": main()
