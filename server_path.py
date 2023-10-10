##############server_path.py##############

'''
export TRANSFORMERS_CACHE=/data-gpu04/nfs/hugging_face_model/
export TORCH_HOME=/data-gpu04/nfs/hugging_face_model/

mkdir $TRANSFORMERS_CACHE

ls -l $TRANSFORMERS_CACHE


python3 app_path.py
'''

import re
import time
import logging
import argsparser
from flask_restx import *
from flask import *

##############llama2_qa.py##############
import time
from transformers import pipeline

print('loading the model')

model_id = "TheBloke/Llama-2-7b-Chat-GPTQ"

pipe = pipeline(
	"text-generation", 
	model = model_id,
	token = "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",
    return_full_text = False,
	max_new_tokens = 512,
	device_map = 6,
	)


#pipe('Hello. My name is ')

def llama2_qa(
    question,
    ):
    try:
        return pipe(f"""Answer this question with an answer of less than 128 words.\nQuestion: {question}\nAnswer:""")[0]['generated_text']
    except:
        return None


def llama2_generate(
    question,
    ):
    try:
        return pipe(question)[0]['generated_text']
    except:
        return None


'''


answer = llama2_qa(
    question = 'What is your name?',
    )

print(answer)

'''
##############llama2_qa.py##############


ns = Namespace(
	'tonomus_llm', 
	description='Tomonus LLM',
	)

args = argsparser.prepare_args()

#############

llama2_qa_parser = ns.parser()
llama2_qa_parser.add_argument('question', type=str, location='json')

llama2_qa_inputs = ns.model(
	'qa', 
		{
			'question': fields.String(example = u"What is the meaning of life?")
		}
	)

@ns.route('/llama2_qa')
class llama2_qa_api(Resource):
	def __init__(self, *args, **kwargs):
		super(llama2_qa_api, self).__init__(*args, **kwargs)
	@ns.expect(llama2_qa_inputs)
	def post(self):		
		start = time.time()
		try:			
			args = llama2_qa_parser.parse_args()	

			output = {}
			output['answer'] = llama2_qa(args['question'])
			output['answer'] = output['answer'].strip()
			output['status'] = 'success'
			output['running_time'] = float(time.time()- start)
			return output, 200
		except Exception as e:
			output = {}
			output['status'] = str(e)
			output['running_time'] = float(time.time()- start)
			return output


#############

llama2_generate_parser = ns.parser()
llama2_generate_parser.add_argument('prompt', type=str, location='json')

llama2_generate_inputs = ns.model(
	'generate', 
		{
			'prompt': fields.String(example = u"Read the following paragraph and answer the question. Paragraph: My name is Amy Pai. Question: What is my name? Answer:")
		}
	)

@ns.route('/llama2_generate')
class llama2_generate_api(Resource):
	def __init__(self, *args, **kwargs):
		super(llama2_generate_api, self).__init__(*args, **kwargs)
	@ns.expect(llama2_generate_inputs)
	def post(self):		
		start = time.time()
		try:			
			args = llama2_generate_parser.parse_args()	

			output = {}
			output['response'] = llama2_generate(args['prompt'])
			output['response'] = output['response'].strip()
			output['status'] = 'success'
			output['running_time'] = float(time.time()- start)
			return output, 200
		except Exception as e:
			output = {}
			output['status'] = str(e)
			output['running_time'] = float(time.time()- start)
			return output


##############server_path.py##############