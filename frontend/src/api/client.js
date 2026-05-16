import axios from "axios";

const http = axios.create({
  baseURL: "/api",
  timeout: 120000,
});

export async function fetchConfig() {
  const { data } = await http.get("/config");
  return data;
}

export async function fetchSamples(language, subset) {
  const { data } = await http.get("/samples", {
    params: { language, subset },
  });
  return data;
}

export async function fetchSample(sampleId, language, subset) {
  const { data } = await http.get(`/sample/${sampleId}`, {
    params: { language, subset },
  });
  return data;
}

export async function postInject(body) {
  const { data } = await http.post("/inject", body);
  return data;
}

export async function postRun(body) {
  const { data } = await http.post("/run", body);
  return data;
}
