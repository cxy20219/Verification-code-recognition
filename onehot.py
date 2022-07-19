import torch
import built_data
def text2Vec(text):
    """
        input:  text
        return: one-hot
    """
    one_hot = torch.zeros(4,len(built_data.captcha_array))
    for i in range(len(text)):
        one_hot[i,built_data.captcha_array.index(text[i])] = 1
    
    return one_hot
def Vec2text(vec):
    """
        input:  one-hot
        return: text
    """
    
    vec = torch.argmax(vec,1)
    text = ""
    for i in vec:
        text += built_data.captcha_array[i]
    return text

if __name__ == "__main__":
    one = text2Vec("123m")
    print(torch.flatten(one))
