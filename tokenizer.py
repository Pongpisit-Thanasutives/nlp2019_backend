from thainlplib import ThaiWordSegmentLabeller
import tensorflow as tf

def nonzero(a):
    return [i for i, e in enumerate(a) if e != 0]

def split(s, indices):
    return [s[i:j] for i,j in zip(indices, indices[1:]+[None])]

class SertisTokenizer:
    def __init__(self):
        self.saved_model_path = 'saved_model'
        self.session = tf.Session()
        model = tf.saved_model.loader.load(self.session, [tf.saved_model.tag_constants.SERVING], self.saved_model_path)
        self.signature = model.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        self.graph = tf.get_default_graph()

    def predict(self, text):
        inputs = [ThaiWordSegmentLabeller.get_input_labels(text)]
        lengths = [len(text)]

        g_inputs = self.graph.get_tensor_by_name(self.signature.inputs['inputs'].name)
        g_lengths = self.graph.get_tensor_by_name(self.signature.inputs['lengths'].name)
        g_training = self.graph.get_tensor_by_name(self.signature.inputs['training'].name)
        g_outputs = self.graph.get_tensor_by_name(self.signature.outputs['outputs'].name)
        y = self.session.run(g_outputs, feed_dict = {g_inputs: inputs, g_lengths: lengths, g_training: False})

        return ' '.join([w for w in split(text, nonzero(y)) if w != ' '])
