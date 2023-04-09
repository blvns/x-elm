import torch
import argparse

def _calc_self_attn(xglm, key):
	key = key.split('.') #decoder, layers, <layer_id>, self_attn, qkv_proj. <weight/bias>
	layer_id = key[2]
	type_name = key[-1]

	xglm_q = xglm['model']['decoder.layers.{}.self_attn.q_proj.{}'.format(layer_id, type_name)]
	xglm_k = xglm['model']['decoder.layers.{}.self_attn.k_proj.{}'.format(layer_id, type_name)]
	xglm_v = xglm['model']['decoder.layers.{}.self_attn.v_proj.{}'.format(layer_id, type_name)]

	#TODO: debug + make sure this is the correct concat order
	#xglm_kvq = torch.cat([xglm_k, xglm_v, xglm_q], dim=0) #ppl = 68483.87
	#xglm_qkv = torch.cat([xglm_q, xglm_k, xglm_v], dim=0) #ppl = 16377.17
	xglm_qvk = torch.cat([xglm_q, xglm_v, xglm_k], dim=0) #ppl = 10663.2

	return xglm_qvk 

def main(args):

	empty = torch.load(args.empty_state)
	xglm = torch.load(args.xglm_state)

	#add XGLM training langs to the empty state
	empty['cfg']['task']['langs'] = xglm['cfg']['task']['langs']

	#put xglm model weights into empty state
	missing_flag = False
	for ms_key in empty['model']:
		if ms_key in xglm['model']:
			assert xglm['model'][ms_key].shape == empty['model'][ms_key].shape
			empty['model'][ms_key] = xglm['model'][ms_key]
			print(ms_key, 'okay')
		#elif 'decoder.layers' in ms_key and 'self_attn' in ms_key:
		#	z = _calc_self_attn(xglm, ms_key)
		#	assert z.shape == empty['model'][ms_key].shape
		#	empty['model'][ms_key] = z
		#	print(ms_key, 'fixed')
		else:
			print(ms_key, 'MISSING')
			missing_flag = True

	if missing_flag: quit()

	torch.save(empty, args.output_state)





if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--empty_state', type=str, required=True)
	parser.add_argument('--xglm_state', type=str, required=True)
	parser.add_argument('--output_state', type=str, required=True)
	args = parser.parse_args()
	print(args)

	main(args)


#EOF