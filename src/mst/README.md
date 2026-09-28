# canoa_grafo# Quantum Walks em Grafos: Árvores Geradoras Clássicas vs. Quânticas

Este repositório documenta um estudo exploratório comparando a construção de
árvores geradoras em grafos $G=(V,E)$ por métodos clássicos e por diferentes
formalismos de **passeios quânticos (quantum walks)**. O objetivo científico
central é responder a uma pergunta:

> *A introdução de mecânica quântica (superposição, interferência de fase) no
> processo de exploração de um grafo produz alguma vantagem mensurável na
> construção de uma árvore geradora de peso mínimo, frente a métodos
> clássicos equivalentes?*

---

## Estrutura do repositório

```
.
├── init/     -> caminhante aleatório clássico (baseline não-ponderado)
└── lab01/    -> comparação Kruskal vs. 5 arquiteturas de quantum walk
```

---

## `init/` — Caminhante aleatório clássico (single-walker)

### O que foi implementado
Um passeio aleatório simples de Markov de primeira ordem sobre $G$: a cada
passo, o caminhante escolhe um vizinho uniformemente ao acaso; se o vizinho
ainda não foi visitado, a aresta percorrida é adicionada à árvore. O processo
termina quando todos os $|V|$ vértices tiverem sido incorporados.

### Papel científico
Serve como **prova de conceito e baseline metodológico**: estabelece que é
possível construir uma árvore geradora válida (não necessariamente mínima)
via um processo estocástico simples, e fixa a interface de dados (grafo,
função de construção, visualização) reutilizada em todo o restante do
trabalho. Não há reivindicação de otimalidade nesta etapa — o grafo usado
aqui não é ponderado.

### Observação metodológica importante
Este método **não gera uma árvore geradora uniforme** (cada árvore possível
do grafo não tem a mesma probabilidade de ocorrer). Para uma distribuição
uniforme sobre árvores geradoras seria necessário o algoritmo de Wilson
(*loop-erased random walk*), não implementado aqui.

---

## `lab01/` — Kruskal (clássico exato) vs. 5 arquiteturas de quantum walk

### Desenho experimental

1. Um grafo ponderado $G=(V,E,w)$ fixo e reprodutível (seed controlada) é
   gerado uma única vez e reutilizado em todos os experimentos, garantindo
   que qualquer diferença de resultado venha do algoritmo, não da instância.
2. **Referência exata**: MST calculada por **Kruskal** com estrutura de
   Union-Find (compressão de caminho), $O(E \log E)$, determinística e
   ótima por construção.
3. **Cinco arquiteturas de quantum walk** foram implementadas para gerar
   árvores geradoras por um processo análogo, porém estocástico e guiado por
   amplitudes de probabilidade complexas:

| Modelo | Espaço de estados | Mecanismo |
|---|---|---|
| **Hadamard** | arcos, $2\|E\|$ | coin generalizado via Transformada de Fourier Discreta (reduz-se exatamente à matriz de Hadamard $\frac{1}{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$ quando o grau do vértice é 2) |
| **Grover** | arcos, $2\|E\|$ | coin de reflexão $2/d \cdot J - I$ em torno do vetor uniforme |
| **Lackadaisical** | arcos + laço próprio, $2\|E\|+\|V\|$ | coin de Grover estendido com um "laço" (self-loop) por vértice, aumentando a probabilidade de permanência local |
| **Szegedy** | pares $(x,y)$, $\|V\|^2$ | quantização de uma cadeia de Markov clássica via duas reflexões $R_B R_A$ sobre os subespaços gerados pela matriz de transição $P$ |
| **Staggered** | vértices, $\|V\|$ | reflexões locais de Grover aplicadas sobre tesselações do grafo (cobertura das arestas por *matchings* máximos), sem registrador de coin |

Cada arquitetura evolui por operadores estritamente unitários (verificação
de norma preservada a cada passo). A cada "tick", o estado é evoluído por
`coin_steps` iterações e a **distribuição marginal de probabilidade sobre os
vértices da fronteira** (vizinhos não visitados de vértices já visitados) é
usada para **amostrar** o próximo vértice a entrar na árvore — a única etapa
de fato aleatória do processo, análoga à medição em um circuito quântico
real. A aresta de conexão é sempre a de menor peso disponível entre o
vértice escolhido e o conjunto já visitado (passo guloso, no espírito do
algoritmo de Prim), isolando o efeito da moeda quântica exclusivamente na
**ordem de exploração**, não na regra de conexão.

### Protocolo de múltiplas execuções
Cada uma das 5 moedas foi executada com **múltiplas seeds e vértices
iniciais diferentes** (inicialmente 3, depois estendido a 4 execuções por
moeda), documentando explicitamente que o mesmo grafo e o mesmo operador
unitário produzem **resultados diferentes a cada medição** — a assinatura
central de qualquer processo quântico, em contraste com o determinismo total
do Kruskal.

### Experimento de controle (variável de confusão)
Para isolar se o ganho observado vinha de fato de efeitos quânticos
(interferência de fase) ou apenas de introduzir aleatoriedade na ordem de
visita, foi executado um **controle clássico**: o mesmo algoritmo guloso de
conexão, mas com o vértice da fronteira escolhido por **sorteio uniforme**
(sem coin, sem amplitude, sem interferência), repetido em $n=500$ execuções
independentes.

### Resultados obtidos (instância de teste, $n=10$–$12$ vértices)

- **Kruskal**: peso ótimo constante em todas as execuções (referência absoluta).
- **Quantum walks (15–20 execuções, 5 moedas)**: pesos tipicamente 15%–160%
  acima do ótimo, com alta variância entre execuções e **sem nenhuma moeda
  dominando consistentemente as demais** nos cenários testados.
- **Controle clássico (Prim com fronteira uniforme, 500 execuções)**: média e
  desvio-padrão estatisticamente indistinguíveis (ou piores) do conjunto de
  resultados quânticos — o *z-score* da média quântica frente à distribuição
  de controle não indicou vantagem em favor do mecanismo quântico.



### Limitações explícitas
- **Szegedy**: a "posição" medida é uma simplificação prática (marginal do
  registro de chegada $y$); a interpretação padrão do formalismo de Szegedy
  é espectral, não posicional.
- **Staggered**: a tesselação foi obtida por uma heurística gulosa de
  *matchings* máximos, não pela construção ótima geral (que admite tesselações
  por cliques maiores).
- **Lackadaisical**: o peso do laço próprio foi fixado em 1; a literatura
  trata esse peso como parâmetro ajustável, otimizável para tarefas de busca.
- Todos os resultados são de uma única instância de grafo pequena
  ($n=10$–$12$); generalização para grafos maiores ou outras topologias não
  foi testada.

---

## Referências conceituais
- Ambainis, A. et al. — *Coined quantum walks on graphs*.
- Wong, T. — *Lackadaisical quantum walks* (2015).
- Szegedy, M. — *Quantum speed-up of Markov chain based algorithms* (2004).
- Portugal, R. — *Staggered quantum walks on graphs* (2016).
- Dürr, C.; Høyer, P. — *A quantum algorithm for finding the minimum* (1996),
  aplicado ao algoritmo de Boruvka para MST quântico com garantia de
  otimalidade.