import {build} from 'esbuild';
import fs from 'node:fs';
const base=process.env.POLISALUD_API_URL;
if(!base||!base.startsWith('https://'))throw new Error('Configura POLISALUD_API_URL con la URL HTTPS de tu backend, sin /api.');
const url=new URL(base);if(url.username||url.password||url.search||url.hash)throw Error('URL inválida.');
fs.rmSync('www',{recursive:true,force:true});fs.cpSync('../web','www',{recursive:true});
fs.writeFileSync('www/config.js','window.POLISALUD_CONFIG='+JSON.stringify({apiBase:base.replace(/\/$/,''),demo:false})+';\n');
console.log('Interfaz móvil preparada. La sesión se conserva solo en memoria; al cerrar la aplicación se vuelve a iniciar sesión.');

await build({entryPoints:['ble-entry.js'],bundle:true,format:'iife',platform:'browser',outfile:'www/ble-native.js',minify:true});
const html=fs.readFileSync('www/index.html','utf8').replace('<script src="live.js" defer>','<script src="ble-native.js" defer></script><script src="live.js" defer>');fs.writeFileSync('www/index.html',html);
