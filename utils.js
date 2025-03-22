import { createHash } from 'node:crypto'
import * as os from "os"

// SHA256 of a buffer, needed for the obtaining SHA256 of a model
function sha256(content) {
    return createHash('sha256').update(content).digest('hex')
}

// Wait
function delay(time) {
    return new Promise(resolve => setTimeout(resolve, time))
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

export { sha256, delay, get_ip_addr };