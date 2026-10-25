import {spawnSync} from 'node:child_process';
import fs from 'node:fs';
const local=process.platform==='win32'?'.venv/Scripts/python.exe':'.venv/bin/python';
const python=process.env.PYTHON||(fs.existsSync(local)?local:'python');
function run(cmd,args){const r=spawnSync(cmd,args,{stdio:'inherit',env:process.env});if(r.error)throw r.error;if(r.status!==0)process.exit(r.status||1);}
for(const f of ['web/app.js','web/live.js','web/config.js'])run(process.execPath,['--check',f]);
run(python,['manage.py','check']);
run(python,['-m','compileall','-q','config','clinical','identity']);
run(python,['manage.py','collectstatic','--noinput']);
console.log('BUILD_OK: Django checks, Python compilation, JavaScript syntax and static assets.');
