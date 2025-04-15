from methods.pina import PINA

def get_model(model_name, args):
    name = model_name.lower()
    options = {
        'pina': PINA,
        }
    return options[name](args)

