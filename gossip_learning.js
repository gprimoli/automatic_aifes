import process from 'node:process'
import { createLibp2p } from 'libp2p'
import { tcp } from '@libp2p/tcp'
import { mplex } from '@libp2p/mplex'
import { noise } from '@chainsafe/libp2p-noise'
import { peerIdFromString } from '@libp2p/peer-id'
import { multiaddr } from 'multiaddr'
import { mdns } from '@libp2p/mdns'
import { pipe } from 'it-pipe'
import toBuffer from 'it-to-buffer'
import pythonBridge from 'python-bridge'

import * as os from "os"
import fs from 'fs'
import { createHash } from 'node:crypto'

import AsyncLock from 'async-lock'

const lock = new AsyncLock()
const path_dir_models = 'models/'
var num_send_to_do = 1
var num_of_known_peers = 0

var peer_id_known_peers = []
var age_local_model = 0
var index_training = 0

const NUM_ROUNDS = 150

function get_position_str(string, subString, index) {
	return string.split(subString, index).join(subString).length
}

function get_ip_addr(){
  const interfaces = os.networkInterfaces();
  for (const name of Object.keys(interfaces)) {
      for (const net of interfaces[name]) {
          // Skip over non-IPv4 and internal (i.e., 127.0.0.1) addresses
          if (net.family === 'IPv4' && !net.internal) {
              return net.address;
          }
      }
  }
  throw new Error('Nessun indirizzo IP valido trovato');
}

function get_peerid_from_multiadd(multiadd){
	return multiadd.substring(get_position_str(multiadd, '/', 6)+1, get_position_str(multiadd, '/', 7))
}

function create_random_str(length) {
    let result = ''
    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    const charactersLength = characters.length
    let counter = 0
    while (counter < length) {
      result += characters.charAt(Math.floor(Math.random() * charactersLength))
      counter += 1
    }
    return result
}

function sha256(content) {  
	return createHash('sha256').update(content).digest('hex')
}

function delay(time) {
  return new Promise(resolve => setTimeout(resolve, time))
}

async function on_model_received ({ stream }) {
  console.log("Metodo on_model_received invocato.")
	const result = await pipe(
		stream,
		async function * (source) {
			for await (const list of source) {
				yield list.subarray()
			}
		},
		toBuffer
    ).finally(() => {
		stream.close()
    })
	
	const model_buff = result.slice(0, result.length-2)
	var random_name_file = create_random_str(10) + '.pt'
	console.log('sha256 del file modello ricevuto: ', sha256(model_buff))
	fs.writeFileSync(path_dir_models + random_name_file, model_buff)
	
	const buffer_age = Buffer.from(result)
	var age_received_model = buffer_age.readUInt16BE(result.length-2)
	console.log('age modello ricevuto: ', age_received_model)
	
	if(index_training < NUM_ROUNDS){
    console.log("Sono al round " + index_training + " su " + NUM_ROUNDS)
		await lock.acquire('key', async() => {
			await python.ex`
			print("loading model received")
  logging.debug("Loading received model.")
	received_model.load_state_dict(torch.load(${path_dir_models} + ${random_name_file}))
			
			print("merging local model with received model")
  logging.debug("Merging local model with received model.")
	merge_models(local_model, int(${age_local_model}), received_model, int(${age_received_model}))
			
			print("training model")	
  logging.debug("Training model.")
	client_update(local_model, opt, train_loader, epoch=epochs)
			
	test_loss, acc = test(local_model, test_loader)
			print('%d-th round, test acc: %0.5f' % (${index_training}, acc))
  logging.debug('%d-th round, test acc: %0.5f' % (${index_training}, acc))
  logging.debug("Training iteration complete.")
			`
			index_training = index_training + 1
			num_send_to_do = num_send_to_do + 1
			age_local_model = age_local_model + 1
      console.log('Index training round: ' + index_training + " Age modello locale: " + age_local_model)
		})
	}
	
	fs.unlinkSync(path_dir_models + random_name_file)
}

var python = pythonBridge({
    python: 'python',
    stdio: ['pipe', process.stdout, process.stderr]
})

if (!fs.existsSync(path_dir_models)){
	fs.mkdirSync(path_dir_models)
}

const my_ip = get_ip_addr()

const createNode = async () => {
	const node = await createLibp2p({
		addresses: {
			listen: ['/ip4/' + my_ip + '/tcp/4000']
		},
		transports: [tcp()],
		streamMuxers: [mplex()],
		connectionEncryption: [noise()],
		peerDiscovery: [mdns()]
	})

	return node
}

const node = await createNode()
console.log('MY ADDRESS: ', node.getMultiaddrs(), '\n')

node.addEventListener('peer:discovery', async(evt) => {
	
	for(let i=0; i < evt.detail.multiaddrs.length; i++){
    //console.log("Evento discovery, iterazione n. " + i)
    //console.log("Numero multiaddr trovati: " + evt.detail.multiaddrs.length + "\n")
    console.log(evt.detail)
		if(evt.detail.multiaddrs[i].toString().includes('tcp')){
      console.log("Peer trovato.")
			//let peerid = get_peerid_from_multiadd(evt.detail.multiaddrs[i].toString())
      let peerid = evt.detail.id
			if(peer_id_known_peers.includes(peerid) == false){
        console.log("Aggiungo il peer.\n")
				peer_id_known_peers.push(peerid)
				await delay(20000)
				num_of_known_peers = num_of_known_peers + 1
        console.log("Peer totali: " + peer_id_known_peers.length + ", Peer conosciuti: " + num_of_known_peers)
			}
		}
	}
  //console.log("Evento discovery, sono fuori dal ciclo.\n")
})


await python.ex`
train_loader = create_train_loader()
if torch.cuda.is_available():
  logging.debug("CUDA disponibile e in uso")
  local_model = MyModel().cuda()
  received_model = MyModel().cuda()
else:
  logging.debug("CUDA non disponibile; utilizzo la CPU.")
  local_model = MyModel().to('cpu')
  received_model = MyModel().to('cpu')
opt = optim.SGD(local_model.parameters(), lr=0.1)
	
test_loader = create_test_loader()
`
console.log("Train loader e test loader creati.\n")

node.handle('/on_model_received', on_model_received)

const my_model_name = 'my_model_' + create_random_str(6) + '.pt'

while(num_of_known_peers < 1){
	await delay(1000)
}

var content_model_file
var age_local_model_to_send
for(let i=0; i< NUM_ROUNDS; i++){
  console.log("Sono nel ciclo per salvare il modello.\n")
	await lock.acquire('key', async() => {
		await python.ex`
		torch.save(local_model.state_dict(), ${path_dir_models} + ${my_model_name})
  logging.debug("Modello salvato.")
		`
		content_model_file = await fs.readFileSync(path_dir_models + my_model_name)
		num_send_to_do = num_send_to_do - 1
		age_local_model_to_send = age_local_model
	})
	let random_peer = Math.floor(Math.random() * num_of_known_peers)
	
	console.log('sha256 del file modello inviato: ', sha256(content_model_file))
	console.log('invio verso', peer_id_known_peers[random_peer], ' random_peer', random_peer)
	
	//const stream = await node.dialProtocol(peerIdFromString(peer_id_known_peers[random_peer]), '/on_model_received')
  const stream = await node.dialProtocol(peer_id_known_peers[random_peer], '/on_model_received')
	const buff_age = Buffer.alloc(2)
	buff_age.writeUInt16BE(age_local_model_to_send)
	const buff_final = Buffer.concat([content_model_file, buff_age], content_model_file.length+2)
	await pipe([buff_final], stream)

	while(num_send_to_do <= 0){
		await delay(2000)
	}
}