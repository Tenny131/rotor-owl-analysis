# Start mit Docker

Dieses Projekt kann mit Docker Compose gestartet werden. Es werden folgende Services gestartet:
- **Streamlit UI**: http://localhost:8501
- **Fuseki (SPARQL)**: http://localhost:3030

---

## Voraussetzungen

- [Docker installieren](https://www.docker.com/products/docker-desktop/)

---

## Start

### Windows (PowerShell)

```powershell
iwr -useb https://raw.githubusercontent.com/Tenny131/rotor-owl-analysis/main/docker-compose.yml | docker compose -f - up --build
```

### Linux / macOS (Terminal)

```bash
curl -fsSL https://raw.githubusercontent.com/Tenny131/rotor-owl-analysis/main/docker-compose.yml | docker compose -f - up --build
```

---

## Im Hintergrund starten

### Windows (PowerShell)

```powershell
iwr -useb https://raw.githubusercontent.com/Tenny131/rotor-owl-analysis/main/docker-compose.yml | docker compose -f - up -d --build
```

### Linux / macOS (Terminal)

```bash
curl -fsSL https://raw.githubusercontent.com/Tenny131/rotor-owl-analysis/main/docker-compose.yml | docker compose -f - up -d --build
```

---

## Container stoppen / starten / neustarten

```bash
docker stop rotor-streamlit rotor-fuseki
docker start rotor-streamlit rotor-fuseki
docker restart rotor-streamlit rotor-fuseki
```

---

## Logs anzeigen

### Streamlit

```bash
docker logs -f rotor-streamlit
```

### Fuseki

```bash
docker logs -f rotor-fuseki
```

---

## Stack herunterfahren

Entfernt Container und Netzwerk (Volumes bleiben erhalten).

### Windows (PowerShell)

```powershell
iwr -useb https://raw.githubusercontent.com/Tenny131/rotor-owl-analysis/main/docker-compose.yml | docker compose -f - down
```

### Linux / macOS (Terminal)

```bash
curl -fsSL https://raw.githubusercontent.com/Tenny131/rotor-owl-analysis/main/docker-compose.yml | docker compose -f - down
```

---

## Alles entfernen (inkl. Volumes)

Entfernt Container, Netzwerk und Volumes.

### Windows (PowerShell)

```powershell
iwr -useb https://raw.githubusercontent.com/Tenny131/rotor-owl-analysis/main/docker-compose.yml | docker compose -f - down -v
```

### Linux / macOS (Terminal)

```bash
curl -fsSL https://raw.githubusercontent.com/Tenny131/rotor-owl-analysis/main/docker-compose.yml | docker compose -f - down -v
```
