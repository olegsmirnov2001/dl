const createForm = document.getElementById('create-form');
const joinForm = document.getElementById('join-form');
const seedInput = document.getElementById('seed');
const roomIdInput = document.getElementById('room-id');
const warning = document.getElementById('warning');

const params = new URLSearchParams(location.search);
const err = params.get('error');
if (err === 'room_not_found') {
  const id = (params.get('id') || '').toUpperCase();
  warning.textContent = id
    ? `Room ${id} not found. It may have expired or never existed.`
    : 'That room was not found.';
  warning.classList.remove('hidden');
  if (id) roomIdInput.value = id;
  history.replaceState({}, '', '/');
}

createForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const body = {};
  if (seedInput.value !== '') body.seed = Number(seedInput.value);
  const r = await fetch('/api/rooms', {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    alert('failed to create room');
    return;
  }
  const data = await r.json();
  window.location.href = `/room/${data.room_id}`;
});

joinForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const id = roomIdInput.value.trim().toUpperCase();
  if (id) window.location.href = `/room/${id}`;
});

const helpToggle = document.getElementById('help-toggle');
const helpPanel = document.getElementById('help-panel');
helpToggle.addEventListener('click', () => {
  const open = helpPanel.classList.toggle('hidden') === false;
  helpToggle.setAttribute('aria-expanded', String(open));
  helpPanel.setAttribute('aria-hidden', String(!open));
  if (open) helpPanel.scrollIntoView({behavior: 'smooth', block: 'nearest'});
});
