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
import { exec } from 'child_process';

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
const my_ip = get_ip_addr()

const root_path = '/app/'

const my_model_file = root_path + 'weights';
const my_training_x_file = root_path + 'dataset/' + my_ip + '/partition_' + my_ip + '.csv';
const my_training_y_file = root_path + 'dataset/' + my_ip + '/partition_' + my_ip + '_y.csv';

const NUM_ROUNDS = 150

function get_position_str(string, subString, index) {
    return string.split(subString, index).join(subString).length
}

function get_ip_addr() {
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

function get_peerid_from_multiadd(multiadd) {
    return multiadd.substring(get_position_str(multiadd, '/', 6) + 1, get_position_str(multiadd, '/', 7))
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

async function on_model_received({ stream }) {
    console.log("Metodo on_model_received invocato.")
    // Model received, so we can start with the flow of training
    const result = await pipe(
        stream,
        async function* (source) {
            for await (const list of source) {
                yield list.subarray()
            }
        },
        toBuffer
    ).finally(() => {
        stream.close()
    })


    // Save the received model in a path
    const model_buff = result.slice(0, result.length - 2) // Without the Age of the model
    console.log('sha256 del file modello ricevuto: ', sha256(model_buff))
    fs.writeFileSync(path_dir_models + sha256(model_buff), model_buff)

    const buffer_age = Buffer.from(result)
    var age_received_model = buffer_age.readUInt16BE(result.length - 2) // Just the age of the model
    console.log('age modello ricevuto: ', age_received_model)

    if (index_training < NUM_ROUNDS) {
        console.log("Sono al round " + index_training + " su " + NUM_ROUNDS)

        // Merge the models
        // Save the resulting model in a path

        // Call the Autoencoder to compute the new model
        let weights_option = "";
        if (fs.existsSync(my_model_file)) weights_option = ` -w ${my_model_file}`;
        let cmd_to_exec = root_path + "automatic_aifes/Autoencoder -l 15,3,1 -a relu,sigmoid -b 32 -e 1000 -i " + my_training_x_file + " -t " + my_training_y_file + weights_option;
        console.log("Executing " + cmd_to_exec);
        exec(cmd_to_exec, (error, stdout, stderr) => {
            if (error) {
                console.error(`Errore: ${error.message}`);
                return;
            }
            if (stderr) {
                console.error(`Stderr: ${stderr}`);
                return;
            }
            console.log(`Output: ${stdout}`);
            console.log('Index training round: ' + index_training + " Age modello locale: " + age_local_model)
        });


    }

    fs.unlinkSync(path_dir_models + sha256(model_buff))
}

var python = pythonBridge({
    python: 'python',
    stdio: ['pipe', process.stdout, process.stderr]
})

if (!fs.existsSync(path_dir_models)) {
    fs.mkdirSync(path_dir_models)
}


// Creazione del nodo, andandomi a prendere un ip disponibile
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

// Listener, se ci sta qualche nuovo nodo allora notificami e aggiungiamolo alla lista
node.addEventListener('peer:discovery', async (evt) => {

    for (let i = 0; i < evt.detail.multiaddrs.length; i++) {
        //console.log("Evento discovery, iterazione n. " + i)
        //console.log("Numero multiaddr trovati: " + evt.detail.multiaddrs.length + "\n")
        console.log(evt.detail)
        if (evt.detail.multiaddrs[i].toString().includes('tcp')) {
            console.log("Peer trovato.")
            //let peerid = get_peerid_from_multiadd(evt.detail.multiaddrs[i].toString())
            let peerid = evt.detail.id
            if (peer_id_known_peers.includes(peerid) == false) {
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



// JUST FOR TEST TO REMOVE IN PROD
let weights_option = "";
if (fs.existsSync(my_model_file)) weights_option = ` -w ${my_model_file}`;
const cmd_to_exec = root_path + "automatic_aifes/Autoencoder -l 15,3,1 -a relu,sigmoid -b 32 -e 1000 -i " + my_training_x_file + " -t " + my_training_y_file + weights_option;
console.log("Executing " + cmd_to_exec);
exec(cmd_to_exec, (error, stdout, stderr) => {
    if (error) {
        console.error(`Errore: ${error.message}`);
        return;
    }
    if (stderr) {
        console.error(`Stderr: ${stderr}`);
        return;
    }
    console.log(`Output: ${stdout}`);
    console.log('Index training round: ' + index_training + " Age modello locale: " + age_local_model)
});


// Register on_model_received
node.handle('/on_model_received', on_model_received)

// Wait till another peer exists
while (num_of_known_peers < 1) {
    await delay(1000)
}


var age_local_model_to_send
for (let i = 0; i < NUM_ROUNDS; i++) {
    console.log("Sono nel ciclo per iniviare il modello aggiornato.\n")

    content_model_file = await fs.readFileSync(my_model_file)
    console.log('sha256 del file modello da inviare: ', sha256(content_model_file))

    let random_peer = Math.floor(Math.random() * num_of_known_peers)
    // Invia verso un random peer (Lo so, è una approssimazione grezza ma è da capire se inviare a tutti i peer)
    console.log('invio verso', peer_id_known_peers[random_peer], ' random_peer', random_peer)

    //const stream = await node.dialProtocol(peerIdFromString(peer_id_known_peers[random_peer]), '/on_model_received')
    const stream = await node.dialProtocol(peer_id_known_peers[random_peer], '/on_model_received')
    const buff_age = Buffer.alloc(2) // Put the age of the model
    buff_age.writeUInt16BE(age_local_model_to_send)
    const buff_final = Buffer.concat([content_model_file, buff_age], content_model_file.length + 2)
    await pipe([buff_final], stream)

    while (num_send_to_do <= 0) {
        await delay(2000)
    }
}