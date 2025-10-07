import { createLibp2p } from 'libp2p'
import { tcp } from '@libp2p/tcp'
import { mplex } from '@libp2p/mplex'
import { noise } from '@chainsafe/libp2p-noise'
import { mdns } from '@libp2p/mdns'
import { pipe } from 'it-pipe'
import toBuffer from 'it-to-buffer'
import { exec } from 'child_process';
import ini from 'ini';
import fs from 'fs';
import { sha256, delay, get_ip_addr } from './utils.js';

import { promisify } from 'util';
var num_send_to_do = 0
var peer_id_known_peers = []
var age_local_model = 0
const my_ip = get_ip_addr()

const root_path = '/app/'




const path_dir_models = root_path + 'models/'
const my_model_file = root_path + 'dataset/' + my_ip + '/save.csv';
const my_training_x_file = root_path + 'dataset/' + my_ip + '/x_train.csv';
const my_training_y_file = root_path + 'dataset/' + my_ip + '/y_train.csv';
const NUM_ROUNDS = 100
const NUM_EPOCHS_PER_ROUND = 1000

const evolution_accuracy = {};

const execp = promisify(exec);

let inFlight = Promise.resolve();

const fileBuffer = fs.readFileSync(my_training_y_file);
const dataset_length = fileBuffer.toString().split("\n").length - 1;


// Create the path_dir_models, used for temp model received
if (!fs.existsSync(path_dir_models)) {
    fs.mkdirSync(path_dir_models)
}

// Function to execute when we receive a model
async function on_model_received({ stream }) {
  console.log("Metodo on_model_received invocato.");

  // 1) Ricezione buffer (modello + 2 byte di age) e chiusura stream
  const result = await pipe(
    stream,
    async function* (source) {
      for await (const list of source) yield list.subarray();
    },
    toBuffer
  ).finally(() => {
    try { stream.close(); } catch {}
  });

  // 2) Split: modello e age
  
  const model_buff = result.slice(0, result.length - 2);
  // Prima di leggere l’età del modello
const buffer_age = Buffer.from(result);  
const age_received_model = buffer_age.readUInt16BE(result.length - 2);
  const sha = sha256(model_buff);
  console.log('sha256 del file modello ricevuto: ', sha);
  console.log('age modello ricevuto aggiornatooooo: ', age_received_model);

  // 3) Persist del file ricevuto (async, atomico)
  try {
    await fs.promises.mkdir(path_dir_models, { recursive: true });
    const receivedPath = path_dir_models + sha;
    await fs.promises.writeFile(receivedPath, model_buff);
    console.log("File modello salvato con successo in:", receivedPath);
  } catch (err) {
    console.error("Errore durante il salvataggio del file modello:", err);
    return; // abort: senza file non possiamo proseguire
  }

  console.log(`Sono al round ${index_training} su ${NUM_ROUNDS}`);

  // 4) Serializza MERGE → TRAIN → LOG usando la coda `inFlight`
  inFlight = inFlight.then(async () => {
    const receivedPath = path_dir_models + sha;
    const toLoadPath = `/app/dataset/${my_ip}/toload.csv`;
    const cfgPath = `/app/automatic_aifes/config_${my_ip}_t.ini`;

    // 4.a) Merge pesi -> toload.csv
    const mergeCmd = `python ${root_path}weights_merge.py "${my_model_file}" "${receivedPath}" --output "${toLoadPath}"`;
    console.log("Eseguo merge:", mergeCmd);
    await execp(mergeCmd);

    // 4.b) Aggiorna INI (abilita/disabilita load)
    const config_ini = ini.parse(fs.readFileSync(cfgPath, 'utf-8'));
    if (fs.existsSync(toLoadPath)) {
      config_ini.general = config_ini.general || {};
      config_ini.general.load = "toload.csv";
    } else {
      if (config_ini.general) delete config_ini.general.load;
    }
    fs.writeFileSync(cfgPath, ini.stringify(config_ini));

    // 4.c) Training con automatic_aifes
    const trainCmd = `${root_path}automatic_aifes/automatic_aifes "${cfgPath}"`;
    console.log("Executing", trainCmd);
    const { stdout } = await execp(trainCmd);

    // 4.d) Parse metriche
    const lines = stdout.split('\n');
    const losses = [];
    const accuracies = [];
    const regex = /\[INFO\] Loss:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*Accuracy:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)/;

    for (const line of lines) {
      const m = line.match(regex);
      if (m) {
        const loss = parseFloat(m[1]);
        const acc  = parseFloat(m[2]);
        if (!Number.isNaN(loss)) losses.push(loss);
        if (!Number.isNaN(acc))  accuracies.push(acc);
      }
    }

    console.log('Losses:', losses);
    console.log('Accuracies:', accuracies);

    // 4.e) Aggiorna evoluzione accuracy (ultimo valore se presente)
    if (accuracies.length > 0) {
      const lastAcc = accuracies[accuracies.length - 1];
      if (evolution_accuracy[my_ip]) {
        evolution_accuracy[my_ip].push(lastAcc);
      } else {
        evolution_accuracy[my_ip] = [lastAcc];
      }
    } else {
      console.warn("Nessuna accuracy trovata nello stdout; non aggiorno la serie.");
    }

    console.log(`Index training round: ${index_training} | Age modello locale: ${age_local_model}`);
    num_send_to_do++;
    age_local_model++;

    console.log(evolution_accuracy);
    await fs.promises.writeFile('result.json', JSON.stringify(evolution_accuracy), 'utf8');

    // 4.f) (Opzionale) cleanup del file ricevuto per evitare accumulo
    try { await fs.promises.unlink(receivedPath); } catch {}
  }).catch((err) => {
    console.error("Errore nella pipeline merge/train:", err);
  });

  // Attendi che questa unit di lavoro finisca prima di restituire
  await inFlight;
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

// JUST FOR TEST TO REMOVE IN PROD, or maybe to mantain for the first training.

let cmd_to_exec = root_path + "automatic_aifes/automatic_aifes \"/app/automatic_aifes/config_" + my_ip + "_t.ini\"";
console.log("Executing " + cmd_to_exec);
await exec(cmd_to_exec, (error, stdout, stderr) => {

    if (error) {
        console.error(`Errore test: ${error.message}`);
        return;
    }
    if (stderr) {
        console.error(`Stderr: ${stderr}`);
        return;
    }
    console.log(`Output: ${stdout}`);
});

// Create a node
const node = await createNode()
console.log('MY ADDRESS: ', node.getMultiaddrs(), '\n')

// Register on_model_received
node.handle('/on_model_received', on_model_received)

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
                console.log("Peer totali: " + peer_id_known_peers.length)
            }
        }
    }
    //console.log("Evento discovery, sono fuori dal ciclo.\n")
})


// Wait till another peer exists
while (peer_id_known_peers.length < 1) {
    await delay(1000)
}

if (my_ip == "172.19.0.2") num_send_to_do = 1 // Just one start sending the model


for (var index_training = 0; index_training < NUM_ROUNDS; index_training++) {
    console.log("Sono nel ciclo per iniviare il modello aggiornato.\n")


    while (num_send_to_do <= 0 || !fs.existsSync(my_model_file)) { // If no one send me its model, then I sleep
        await delay(2000)
    }

    let random_peer = Math.floor(Math.random() * peer_id_known_peers.length)
    // Invia verso un random peer (Lo so, è una approssimazione grezza ma è da capire se inviare a tutti i peer) o se verso solo chi me l'ha inviato
    console.log('invio verso', peer_id_known_peers[random_peer], ' random_peer', random_peer)

    //const stream = await node.dialProtocol(peerIdFromString(peer_id_known_peers[random_peer]), '/on_model_received')
    const stream = await node.dialProtocol(peer_id_known_peers[random_peer], '/on_model_received') // Open the stream

    let content_model_file = await fs.readFileSync(root_path + 'dataset/' + my_ip + '/save.csv')
    console.log('sha256 del file modello da inviare: ', sha256(content_model_file))

    const buff_age = Buffer.alloc(2) // Put the age of the model
    buff_age.writeUInt16BE(age_local_model)

    const buff_final = Buffer.concat([content_model_file, buff_age], content_model_file.length + 2)
    await pipe([buff_final], stream) // Send it

    num_send_to_do--; // Decrease the send to do, in case a new model arrives then we will increase it again.

}