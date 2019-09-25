from __future__ import print_function, unicode_literals
#from PyInquirer import style_from_dict, Token, prompt, Separator


def checkbox(name, options, msg=None):
    style = style_from_dict({
    Token.Separator: '#27ae60',
    Token.QuestionMark: '#00d700 bold',
    Token.Selected: '#27ae60',  # default
    Token.Pointer: '#00d700 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#00d700 bold',
    Token.Question: '',
    })

    choices = [Separator('= {} ='.format(name))]
    for option in options:
        choices.append({'name': option, 'checked': True})

    if msg is None:
        msg = 'Select {}'.format(name)

    questions = [
        {
            'type': 'checkbox',
            'message': msg,
            'name': name,
            'choices': choices,
            'validate': lambda answer: 'You must choose at least one option.' \
                if len(answer) == 0 else True
        }
    ]

    return prompt(questions, style=style)[name]
