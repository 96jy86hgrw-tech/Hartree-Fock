/* =========================================================
   Hartree–Fock project site
   ========================================================= */

/* ---------- Section switching (your original logic, tidied) ---------- */
function showSection(sectionId, linkEl) {
  document.querySelectorAll("main section").forEach((sec) => (sec.hidden = true));
  const target = document.getElementById(sectionId);
  if (target) target.hidden = false;

  // update sidebar active state (skip when coming from the table itself)
  document.querySelectorAll(".nav-link").forEach((a) => a.classList.remove("active"));
  if (linkEl && linkEl.classList.contains("nav-link")) linkEl.classList.add("active");

  document.querySelector(".content").scrollTo({ top: 0, behavior: "smooth" });
}

/* ---------- Atom data: edit these freely ---------- */
const ATOMS = {
  Li: {
    name: "Lithium",
    config: "1s² 2s¹",
    shells: [2, 1],
    energy: "−7.4285",
    desc: "",
    blurb:
      "Open-shell, so it uses the unrestricted (UHF) method: two density matrices — one per spin — give two coupled Fock equations sharing the total Coulomb potential but carrying separate exchange terms.",
    code: `import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

# --- Parameters ---
Z = 3
N = 800
R = 12.5
dr = R / (N + 1)
r = dr * np.arange(1, N + 1)
# Radial grid starting at dr rather than 0 to avoid the singularity

# --- Functions ---
def build_T_operator(N, dr):
    main = np.full(N, -2.0)
    off = np.ones(N - 1)
    D2 = diags([off, main, off], offsets=[-1, 0, 1], shape=(N, N)) / (dr**2)
    return (-0.5) * D2

def normalize_u(u, r):
    norm = np.sqrt(np.trapezoid(u**2, r))
    return u / norm

def build_r12_matrix(r):
    return 1.0 / np.maximum.outer(r, r)

def build_p_a_matrix(u1, u2):
    return np.outer(u1, u1) + np.outer(u2, u2)

def build_p_b_matrix(u1):
    return np.outer(u1, u1)

def calculate_J(P, r12_kernel, dr):
    rho = np.diag(P)
    J_vec = np.dot(r12_kernel, rho) * dr
    return diags(J_vec, 0)

def calculate_K_UHF(P, r12_kernel, dr):
    return P * r12_kernel * dr

# --- SCF Setup ---
T = build_T_operator(N, dr)
V_nuc = diags(-Z / r, 0)
kernel = build_r12_matrix(r)

# Initial Guess (Hydrogenic)
u1sa = normalize_u(r * (Z)**1.5 * 2 * np.exp(-Z * r), r)
u1sb = normalize_u(r * (Z)**1.5 * 2 * np.exp(-Z * r), r)
u2sa = normalize_u(r * (Z/2)**1.5 * (1 - Z*r/2) * np.exp(-Z*r/2), r)

MAX_ITER = 50
TOLERANCE = 1e-7
E_1sa_history, E_1sb_history, E_2sa_history = ([] for _ in range(3))

# --- Main SCF Loop ---
for i in range(MAX_ITER):
    Pa = build_p_a_matrix(u1sa, u2sa)
    Pb = build_p_b_matrix(u1sb)
    P = Pa + Pb

    J_mat = calculate_J(P, kernel, dr)
    K_mat_a = calculate_K_UHF(Pa, kernel, dr)
    K_mat_b = calculate_K_UHF(Pb, kernel, dr)

    Fock_a = T + V_nuc + J_mat - K_mat_a
    eigenvalues_a, eigenvectors_a = eigsh(Fock_a, k=2, which='SA')

    Fock_b = T + V_nuc + J_mat - K_mat_b
    eigenvalues_b, eigenvectors_b = eigsh(Fock_b, k=1, which='SA')

    new_u1sa = normalize_u(eigenvectors_a[:, 0], r)
    new_u2sa = normalize_u(eigenvectors_a[:, 1], r)
    new_u1sb = normalize_u(eigenvectors_b[:, 0], r)

    e1sa, e2sa = eigenvalues_a[0], eigenvalues_a[1]
    e1sb = eigenvalues_b[0]

    delta_e = abs(e1sa - E_1sa_history[-1]) if E_1sa_history else 999
    E_1sa_history.append(e1sa)
    E_1sb_history.append(e1sb)
    E_2sa_history.append(e2sa)

    if delta_e < TOLERANCE:
        H_core = T + V_nuc
        E_core = (np.dot(u1sa, H_core.dot(u1sa)) +
                  np.dot(u1sb, H_core.dot(u1sb)) +
                  np.dot(u2sa, H_core.dot(u2sa))) * dr
        E_total = 0.5 * (E_core + e1sa + e1sb + e2sa)
        break

    u1sa, u1sb, u2sa = new_u1sa, new_u1sb, new_u2sa`,
  },

  Be: {
    name: "Beryllium",
    config: "1s² 2s²",
    shells: [2, 2],
    energy: "−14.5673",
    desc: "",
    blurb:
      "Closed-shell, so it uses the restricted (RHF) method: a single density matrix and one Fock matrix F = T + V + J − K, with each spatial orbital doubly occupied.",
    code: `import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

# --- Parameters ---
Z = 4
N = 800
R = 12.5
dr = R / (N + 1)
r = dr * np.arange(1, N + 1)

# --- Functions ---
def build_T_operator(N, dr):
    main = np.full(N, -2.0)
    off = np.ones(N - 1)
    D2 = diags([off, main, off], offsets=[-1, 0, 1], shape=(N, N)) / (dr**2)
    return (-0.5) * D2

def normalize_u(u, r):
    norm = np.sqrt(np.trapezoid(u**2, r))
    return u / norm

def build_r12_matrix(r):
    return 1.0 / np.maximum.outer(r, r)

def build_p_matrix(u1, u2):          # P matrix (see textbook)
    return 2 * (np.outer(u1, u1) + np.outer(u2, u2))

def calculate_J(P, r12_kernel, dr):
    rho = np.diag(P)
    J_vec = np.dot(r12_kernel, rho) * dr
    return diags(J_vec, 0)

def calculate_K(P, r12_kernel, dr):  # simplified exchange (closed shell)
    return 0.5 * P * r12_kernel * dr

# --- SCF Setup ---
T = build_T_operator(N, dr)
V_nuc = diags(-Z / r, 0)
kernel = build_r12_matrix(r)

# Initial guess
u1s = normalize_u(r, r)
u2s = normalize_u(r, r)

MAX_ITER = 50
TOLERANCE = 1e-7
E_1s_history, E_2s_history = [], []

# --- Main SCF Loop ---
for i in range(MAX_ITER):
    P = build_p_matrix(u1s, u2s)

    J_mat = calculate_J(P, kernel, dr)
    K_mat = calculate_K(P, kernel, dr)

    Fock = T + V_nuc + J_mat - K_mat
    eigenvalues, eigenvectors = eigsh(Fock, k=2, which='SA')

    new_u1s = normalize_u(eigenvectors[:, 0], r)
    new_u2s = normalize_u(eigenvectors[:, 1], r)

    e1, e2 = eigenvalues[0], eigenvalues[1]

    delta_e = abs(e1 - E_1s_history[-1]) if E_1s_history else 999
    E_1s_history.append(e1)
    E_2s_history.append(e2)

    if delta_e < TOLERANCE:
        # Total energy: 0.5 * (E_core + sum of orbital energies)
        H_core = T + V_nuc
        E_core_1s = np.dot(u1s, H_core.dot(u1s)) * dr
        E_core_2s = np.dot(u2s, H_core.dot(u2s)) * dr
        E_core_total = 2 * E_core_1s + 2 * E_core_2s
        sum_orbital_energies = 2 * e1 + 2 * e2
        E_total = 0.5 * (E_core_total + sum_orbital_energies)

        print(f"Final Results for Beryllium (Z={Z}):")
        print(f"1s Orbital Energy: {e1:.6f} Ha")
        print(f"2s Orbital Energy: {e2:.6f} Ha")
        print(f"Total HF Energy:   {E_total:.6f} Ha")
        break

    u1s, u2s = new_u1s, new_u2s`,
  },

  B: {
    name: "Boron",
    config: "1s² 2s² 2p¹",
    shells: [2, 3],
    energy: "−24.4857",
    desc: "",
    blurb:
      "Introduces the 2p orbital, breaking spherical symmetry. A third density matrix is added for the p-orbital due to the angular momentum term in the hamiltonian, and the dipole term in the multipole expansion (1/3 angular factor) enters the exchange calculation. Three Fock equations 'seeing' different potentials follow:.",
    code: `import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

# --- Parameters ---
Z = 5
N = 800
R = 12.5
dr = R / (N + 1)
r = dr * np.arange(1, N + 1)

# --- Functions ---
def build_T_operator(N, dr):
    main = np.full(N, -2.0)
    off = np.ones(N - 1)
    D2 = diags([off, main, off], offsets=[-1, 0, 1], shape=(N, N)) / (dr**2)
    return (-0.5) * D2

def normalize_u(u, r):
    norm = np.sqrt(np.trapezoid(u**2, r))
    return u / norm

def build_r12_matrix(r):              # monopole kernel
    return 1.0 / np.maximum.outer(r, r)

def build_r12_matrix1(r):             # dipole kernel
    r_less = np.minimum.outer(r, r)
    r_greater = np.maximum.outer(r, r)
    return r_less / (r_greater**2)

def build_p_a_matrix(u1, u2):
    return np.outer(u1, u1) + np.outer(u2, u2)

def build_p_b_matrix(u1, u2):
    return np.outer(u1, u1) + np.outer(u2, u2)

def build_p_a_matrix_2p(u3):
    return np.outer(u3, u3)

def calculate_J(P, r12_kernel, dr):
    rho = np.diag(P)
    J_vec = np.dot(r12_kernel, rho) * dr
    return diags(J_vec, 0)

def calculate_K_UHF_0(P, r12_kernel, dr):
    return P * r12_kernel * dr

def calculate_K_UHF_1(P, r12_kernel_1, dr):
    return P * r12_kernel_1 * 1/3 * dr   # 1/3 angular factor (s-p)

# --- SCF Setup ---
T = build_T_operator(N, dr)
V_nuc = diags(-Z / r, 0)
V_centrifugal = diags(1.0 / (r**2), 0)
kernel_0 = build_r12_matrix(r)
kernel_1 = build_r12_matrix1(r)

# Initial guess (grid; converges fast)
u1sa = normalize_u(r, r)
u1sb = normalize_u(r, r)
u2sa = normalize_u(r, r)
u2sb = normalize_u(r, r)
u2pa = normalize_u(r, r)

MAX_ITER = 50
TOLERANCE = 1e-7
E_1sa_history, E_1sb_history, E_2sa_history, E_2sb_history, E_2pa_history = ([] for _ in range(5))

# --- Main SCF Loop ---
for i in range(MAX_ITER):
    Pa = build_p_a_matrix(u1sa, u2sa)
    Pb = build_p_b_matrix(u1sb, u2sb)
    P2p = build_p_a_matrix_2p(u2pa)
    P = Pa + Pb + P2p

    J_mat = calculate_J(P, kernel_0, dr)

    # Each spin/shell sees a different exchange combination
    K_mat_sa = calculate_K_UHF_0(Pa, kernel_0, dr) + calculate_K_UHF_1(P2p, kernel_1, dr)
    K_mat_sb = calculate_K_UHF_0(Pb, kernel_0, dr)
    K_mat_pa = calculate_K_UHF_1(Pa, kernel_1, dr) + calculate_K_UHF_0(P2p, kernel_0, dr)

    Fock_sa = T + V_nuc + J_mat - K_mat_sa
    eigenvalues_sa, eigenvectors_sa = eigsh(Fock_sa, k=2, which='SA')

    Fock_sb = T + V_nuc + J_mat - K_mat_sb
    eigenvalues_sb, eigenvectors_sb = eigsh(Fock_sb, k=2, which='SA')

    Fock_pa = T + V_nuc + V_centrifugal + J_mat - K_mat_pa
    eigenvalues_pa, eigenvectors_pa = eigsh(Fock_pa, k=1, which='SA')

    new_u1sa = normalize_u(eigenvectors_sa[:, 0], r)
    new_u2sa = normalize_u(eigenvectors_sa[:, 1], r)
    new_u1sb = normalize_u(eigenvectors_sb[:, 0], r)
    new_u2sb = normalize_u(eigenvectors_sb[:, 1], r)
    new_u2pa = normalize_u(eigenvectors_pa[:, 0], r)

    e1sa, e2sa = eigenvalues_sa[0], eigenvalues_sa[1]
    e1sb, e2sb = eigenvalues_sb[0], eigenvalues_sb[1]
    e2pa = eigenvalues_pa[0]

    delta_e = abs(e1sa - E_1sa_history[-1]) if E_1sa_history else 999
    E_1sa_history.append(e1sa)
    E_1sb_history.append(e1sb)
    E_2sa_history.append(e2sa)
    E_2sb_history.append(e2sb)
    E_2pa_history.append(e2pa)

    if delta_e < TOLERANCE:
        H_core = T + V_nuc
        H_core_p = T + V_nuc + V_centrifugal
        E_core_1sa = np.dot(u1sa, H_core.dot(u1sa)) * dr
        E_core_1sb = np.dot(u1sb, H_core.dot(u1sb)) * dr
        E_core_2sa = np.dot(u2sa, H_core.dot(u2sa)) * dr
        E_core_2sb = np.dot(u2sb, H_core.dot(u2sb)) * dr
        E_core_2pa = np.dot(u2pa, H_core_p.dot(u2pa)) * dr
        E_core_total = E_core_1sa + E_core_1sb + E_core_2sa + E_core_2sb + E_core_2pa
        sum_orbital_energies = e1sa + e1sb + e2sa + e2sb + e2pa
        E_total = 0.5 * (E_core_total + sum_orbital_energies)

        print(f"Final Results for Boron (Z={Z}):")
        print(f"Total HF Energy: {E_total:.6f} Ha")
        break

    u1sa, u1sb, u2sa, u2sb, u2pa = new_u1sa, new_u1sb, new_u2sa, new_u2sb, new_u2pa`,
  },

  C: {
    name: "Carbon",
    config: "1s² 2s² 2p²",
    shells: [2, 4],
    energy: "−37.5734",
    desc: "",
    blurb:
      "Adds a second 2p electron (parallel spin, by Hund's rule), introducing a p–p exchange term with a 2/25 angular factor.",
    code: `import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

# --- Parameters ---
Z = 6
N = 800
R = 12.5
dr = R / (N + 1)
r = dr * np.arange(1, N + 1)

# --- Functions ---
def build_T_operator(N, dr):
    main = np.full(N, -2.0)
    off = np.ones(N - 1)
    D2 = diags([off, main, off], offsets=[-1, 0, 1], shape=(N, N)) / (dr**2)
    return (-0.5) * D2

def normalize_u(u, r):
    norm = np.sqrt(np.trapezoid(u**2, r))
    return u / norm

def build_r12_matrix(r):              # monopole
    return 1.0 / np.maximum.outer(r, r)

def build_r12_matrix1(r):             # dipole
    r_less = np.minimum.outer(r, r)
    r_greater = np.maximum.outer(r, r)
    return r_less / (r_greater**2)

def build_r12_matrix2(r):             # quadrupole-like (p-p)
    r_less = np.minimum.outer(r, r)
    r_greater = np.maximum.outer(r, r)
    return r_less**2 / (r_greater**3)

def build_p_a_matrix(u1, u2):
    return np.outer(u1, u1) + np.outer(u2, u2)

def build_p_b_matrix(u1, u2):
    return np.outer(u1, u1) + np.outer(u2, u2)

def build_p_a_matrix_2p(u3):
    return np.outer(u3, u3)

def calculate_J(P, r12_kernel, dr):
    rho = np.diag(P)
    J_vec = np.dot(r12_kernel, rho) * dr
    return diags(J_vec, 0)

def calculate_K_UHF_0(P, r12_kernel, dr):
    return P * r12_kernel * dr

def calculate_K_UHF_1(P, r12_kernel_1, dr):
    return P * r12_kernel_1 * dr

def calculate_K_UHF_2(P, r12_kernel_2, dr):
    return P * r12_kernel_2 * dr

# --- SCF Setup ---
T = build_T_operator(N, dr)
V_nuc = diags(-Z / r, 0)
V_centrifugal = diags(1.0 / (r**2), 0)
kernel_0 = build_r12_matrix(r)
kernel_1 = build_r12_matrix1(r)
kernel_2 = build_r12_matrix2(r)

# Initial guess (hydrogenic)
u1sa = normalize_u(2 * Z**(3/2) * np.exp(-Z * r), r)
u1sb = normalize_u(2 * Z**(3/2) * np.exp(-Z * r), r)
u2sa = normalize_u((Z / 2)**(3/2) * (2 - Z * r) * np.exp(-Z * r / 2), r)
u2sb = normalize_u((Z / 2)**(3/2) * (2 - Z * r) * np.exp(-Z * r / 2), r)
u2pa = normalize_u((Z / 2)**(3/2) * (Z * r / np.sqrt(3)) * np.exp(-Z * r / 2), r)

MAX_ITER = 50
TOLERANCE = 1e-7
E_1sa_history, E_1sb_history, E_2sa_history, E_2sb_history, E_2pa_history = ([] for _ in range(5))

# --- Main SCF Loop ---
for i in range(MAX_ITER):
    Pa = build_p_a_matrix(u1sa, u2sa)
    Pb = build_p_b_matrix(u1sb, u2sb)
    P2p = build_p_a_matrix_2p(u2pa)
    P = Pa + Pb + 2 * P2p             # two parallel-spin 2p electrons (Hund)

    J_mat = calculate_J(P, kernel_0, dr)

    K_mat_sa = calculate_K_UHF_0(Pa, kernel_0, dr) + 2/3 * calculate_K_UHF_1(P2p, kernel_1, dr)
    K_mat_sb = calculate_K_UHF_0(Pb, kernel_0, dr)
    K_mat_pa = (1/3 * calculate_K_UHF_1(Pa, kernel_1, dr)
                + calculate_K_UHF_0(P2p, kernel_0, dr)
                + 2/25 * calculate_K_UHF_2(P2p, kernel_2, dr))

    Fock_sa = T + V_nuc + J_mat - K_mat_sa
    eigenvalues_sa, eigenvectors_sa = eigsh(Fock_sa, k=2, which='SA')

    Fock_sb = T + V_nuc + J_mat - K_mat_sb
    eigenvalues_sb, eigenvectors_sb = eigsh(Fock_sb, k=2, which='SA')

    Fock_pa = T + V_nuc + V_centrifugal + J_mat - K_mat_pa
    eigenvalues_pa, eigenvectors_pa = eigsh(Fock_pa, k=1, which='SA')

    new_u1sa = normalize_u(eigenvectors_sa[:, 0], r)
    new_u2sa = normalize_u(eigenvectors_sa[:, 1], r)
    new_u1sb = normalize_u(eigenvectors_sb[:, 0], r)
    new_u2sb = normalize_u(eigenvectors_sb[:, 1], r)
    new_u2pa = normalize_u(eigenvectors_pa[:, 0], r)

    e1sa, e2sa = eigenvalues_sa[0], eigenvalues_sa[1]
    e1sb, e2sb = eigenvalues_sb[0], eigenvalues_sb[1]
    e2pa = eigenvalues_pa[0]

    delta_e = abs(e1sa - E_1sa_history[-1]) if E_1sa_history else 999
    E_1sa_history.append(e1sa)
    E_1sb_history.append(e1sb)
    E_2sa_history.append(e2sa)
    E_2sb_history.append(e2sb)
    E_2pa_history.append(e2pa)

    if delta_e < TOLERANCE:
        H_core = T + V_nuc
        H_core_p = T + V_nuc + V_centrifugal
        E_core_1sa = np.dot(u1sa, H_core.dot(u1sa)) * dr
        E_core_1sb = np.dot(u1sb, H_core.dot(u1sb)) * dr
        E_core_2sa = np.dot(u2sa, H_core.dot(u2sa)) * dr
        E_core_2sb = np.dot(u2sb, H_core.dot(u2sb)) * dr
        E_core_2pa = np.dot(u2pa, H_core_p.dot(u2pa)) * dr
        E_core_total = E_core_1sa + E_core_1sb + E_core_2sa + E_core_2sb + 2 * E_core_2pa
        sum_orbital_energies = e1sa + e1sb + e2sa + e2sb + 2 * e2pa
        E_total = 0.5 * (E_core_total + sum_orbital_energies)

        print(f"Final Results for Carbon (Z={Z}):")
        print(f"Total HF Energy: {E_total:.6f} Ha")
        break

    u1sa, u1sb, u2sa, u2sb, u2pa = new_u1sa, new_u1sb, new_u2sa, new_u2sb, new_u2pa`,
  },
};

/* ---------- Periodic table layout ----------
   [symbol, column, row, category]. Only the four in ATOMS are clickable. */
const TABLE = [
  ["H", 1, 1, "nonmetal"], ["He", 18, 1, "noble"],
  ["Li", 1, 2, "alkali"], ["Be", 2, 2, "alkaline"],
  ["B", 13, 2, "metalloid"], ["C", 14, 2, "nonmetal"], ["N", 15, 2, "nonmetal"],
  ["O", 16, 2, "nonmetal"], ["F", 17, 2, "halogen"], ["Ne", 18, 2, "noble"],
  ["Na", 1, 3, "alkali"], ["Mg", 2, 3, "alkaline"], ["Al", 13, 3, "post"],
  ["Si", 14, 3, "metalloid"], ["P", 15, 3, "nonmetal"], ["S", 16, 3, "nonmetal"],
  ["Cl", 17, 3, "halogen"], ["Ar", 18, 3, "noble"],
];

function buildTable() {
  const grid = document.getElementById("periodic-table");
  TABLE.forEach(([sym, col, row, cat]) => {
    const live = ATOMS.hasOwnProperty(sym);
    const cell = document.createElement(live ? "button" : "div");
    cell.className = "elem" + (live ? " live" : "");
    cell.textContent = sym;
    cell.style.gridColumn = col;
    cell.style.gridRow = row;
    if (live) cell.addEventListener("click", () => viewElement(sym));
    grid.appendChild(cell);
  });
}

/* ---------- Open an atom's detail view ---------- */
function viewElement(sym) {
  const a = ATOMS[sym];
  document.getElementById("el-name").textContent = a.name;
  document.getElementById("el-config").textContent = a.config;
  document.getElementById("el-blurb").textContent = a.blurb;
  document.getElementById("el-energy").textContent = a.energy + " Eₕ";
  document.getElementById("el-desc").textContent = a.desc || "";

  const codeEl = document.getElementById("el-code");
  codeEl.textContent = a.code;
  if (window.Prism) Prism.highlightElement(codeEl);

  const dia = document.getElementById("el-diagram");
  dia.innerHTML = "";
  buildBohr(dia, a.shells);

  showSection("element-detail-section", null);
}

/* ---------- Bohr diagram from a shells array, e.g. [2, 1] ---------- */
function buildBohr(el, shells) {
  const size = 200, c = size / 2;
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${size} ${size}`);
  svg.setAttribute("class", "bohr");

  shells.forEach((count, i) => {
    const radius = 30 + i * 34;

    const ring = document.createElementNS(ns, "circle");
    ring.setAttribute("cx", c); ring.setAttribute("cy", c);
    ring.setAttribute("r", radius);
    ring.setAttribute("fill", "none");
    ring.setAttribute("stroke", "var(--line)");
    ring.setAttribute("stroke-width", "1.5");
    svg.appendChild(ring);

    const g = document.createElementNS(ns, "g");
    g.setAttribute("class", "bohr-shell");
    g.style.setProperty("--dur", `${10 + i * 6}s`);
    g.style.transformOrigin = `${c}px ${c}px`;

    for (let k = 0; k < count; k++) {
      const angle = (2 * Math.PI * k) / count;
      const e = document.createElementNS(ns, "circle");
      e.setAttribute("cx", c + radius * Math.cos(angle));
      e.setAttribute("cy", c + radius * Math.sin(angle));
      e.setAttribute("r", "4.5");
      e.setAttribute("fill", "var(--orange)");
      g.appendChild(e);
    }
    svg.appendChild(g);
  });

  const nuc = document.createElementNS(ns, "circle");
  nuc.setAttribute("cx", c); nuc.setAttribute("cy", c);
  nuc.setAttribute("r", "11");
  nuc.setAttribute("fill", "var(--orange-deep)");
  svg.appendChild(nuc);

  el.appendChild(svg);
}

/* ---------- Copy code to clipboard ---------- */
function copyCode() {
  const code = document.getElementById("el-code").textContent;
  const btn = document.getElementById("copy-btn");
  navigator.clipboard.writeText(code).then(() => {
    btn.textContent = "Copied ✓";
    btn.classList.add("copied");
    setTimeout(() => {
      btn.textContent = "Copy";
      btn.classList.remove("copied");
    }, 1600);
  }).catch(() => {
    btn.textContent = "Press ⌘C";
  });
}

/* ---------- Init ---------- */
buildTable();