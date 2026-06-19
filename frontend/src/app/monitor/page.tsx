'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  Handle,
  Position,
  BackgroundVariant,
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

// ─── Types ───────────────────────────────────────────────────────────────────
type NodeStatus = 'idle' | 'running' | 'completed' | 'error' | 'skipped';
type EventType =
  | 'query_start'
  | 'node_enter'
  | 'tool_call'
  | 'tool_result'
  | 'node_exit'
  | 'query_complete'
  | 'error'
  | 'ping';

interface NodeData extends Record<string, unknown> {
  label: string;
  nodeType: string;
  status: NodeStatus;
  subtitle?: string;
}

interface WorkflowEvent {
  type: EventType;
  node?: string;
  tool?: string;
  args?: Record<string, unknown>;
  output_preview?: string;
  query?: string;
  thread_id?: string;
  timestamp?: string;
  log?: string;
}

interface LogEntry {
  id: number;
  time: string;
  type: EventType | string;
  message: string;
}

// ─── Status visual config ────────────────────────────────────────────────────
const STATUS: Record<NodeStatus, { border: string; bg: string; text: string; dot: string }> = {
  idle:      { border: '#2D3748', bg: '#161B27', text: '#4A5568', dot: '#2D3748'  },
  running:   { border: '#4299E1', bg: '#1A2E4A', text: '#90CDF4', dot: '#4299E1'  },
  completed: { border: '#38A169', bg: '#1A3030', text: '#68D391', dot: '#38A169'  },
  error:     { border: '#E53E3E', bg: '#3D1515', text: '#FC8181', dot: '#E53E3E'  },
  skipped:   { border: '#4A5568', bg: '#1A202C', text: '#718096', dot: '#4A5568'  },
};

// ─── Node type visual config ─────────────────────────────────────────────────
const NODE_TYPE: Record<string, { icon: string; accent: string }> = {
  start:  { icon: '▶',  accent: '#38A169' },
  llm:    { icon: '⬡',  accent: '#9F7AEA' },
  router: { icon: '◈',  accent: '#ECC94B' },
  tool:   { icon: '◎',  accent: '#4FD1C5' },
  end:    { icon: '■',  accent: '#FC8181' },
};

// ─── Tool → node/edge mapping ─────────────────────────────────────────────────
const TOOL_MAP: Record<string, { nodeId: string; fwdEdge: string; backEdge: string }> = {
  rag_search:              { nodeId: 'ragSearch',     fwdEdge: 'tr-rag',    backEdge: 'rag-chat'    },
  get_kpis:                { nodeId: 'getKpis',       fwdEdge: 'tr-kpi',    backEdge: 'kpi-chat'    },
  generate_report_sections:{ nodeId: 'genReport',     fwdEdge: 'tr-report', backEdge: 'report-chat' },
};

// ─── Edge helpers ─────────────────────────────────────────────────────────────
const baseEdge = (color = '#2D3748') => ({
  animated: false,
  style: { stroke: color, strokeWidth: 1.5 },
  markerEnd: { type: MarkerType.ArrowClosed, color },
});

const activeEdge = () => ({
  animated: true,
  style: { stroke: '#4299E1', strokeWidth: 2 },
  markerEnd: { type: MarkerType.ArrowClosed, color: '#4299E1' },
});

function applyEdge(e: any, active: boolean) {
  return active ? { ...e, ...activeEdge() } : { ...e, ...baseEdge() };
}

// ─── Custom WorkflowNode ──────────────────────────────────────────────────────
function WorkflowNode({ data }: { data: any }) {
  const { label, subtitle, nodeType, status = 'idle' } = data as {
    label: string; subtitle?: string; nodeType: string; status: NodeStatus;
  };
  const s = STATUS[status];
  const t = NODE_TYPE[nodeType] ?? { icon: '◌', accent: '#4A5568' };
  const isRunning = status === 'running';

  return (
    <>
      <style>{`
        @keyframes borderPulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(236, 201, 75, 0.5); }
          50%       { box-shadow: 0 0 0 7px rgba(236, 201, 75, 0); }
        }
      `}</style>
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: s.dot, border: `1px solid ${s.border}`, width: 8, height: 8, left: -5 }}
      />

      <div style={{
        background: s.bg,
        borderTop: `1.5px solid ${isRunning ? '#ECC94B' : s.border}`,
        borderRight: `1.5px solid ${isRunning ? '#ECC94B' : s.border}`,
        borderBottom: `1.5px solid ${isRunning ? '#ECC94B' : s.border}`,
        borderLeft: `3px solid ${isRunning ? '#ECC94B' : t.accent}`,
        borderRadius: '10px',
        padding: '10px 14px',
        minWidth: '170px',
        maxWidth: '230px',
        transition: 'all 0.35s ease',
        animation: isRunning ? 'borderPulse 1.4s ease-in-out infinite' : 'none',
        cursor: 'default',
        userSelect: 'none',
      }}>
        {/* Row: icon · label · status dot */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 13, color: t.accent, lineHeight: 1 }}>{t.icon}</span>
          <span style={{
            fontSize: '0.75rem', fontWeight: 700, color: s.text,
            fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.02em',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1,
          }}>{label}</span>
          <span style={{
            width: 7, height: 7, borderRadius: '50%',
            background: s.dot,
            boxShadow: isRunning ? `0 0 8px ${s.dot}` : 'none',
            flexShrink: 0,
            transition: 'box-shadow 0.3s',
          }} />
        </div>
        {/* Subtitle */}
        {subtitle && (
          <div style={{
            fontSize: '0.62rem', color: '#4A5568',
            fontFamily: "'JetBrains Mono', monospace",
            marginTop: 4, marginLeft: 21,
          }}>{subtitle}</div>
        )}
        {/* Status badge */}
        {status !== 'idle' && (
          <div style={{
            marginTop: 6, marginLeft: 21,
            fontSize: '0.58rem',
            fontFamily: "'JetBrains Mono', monospace",
            color: t.accent,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
          }}>{status}</div>
        )}
      </div>

      <Handle
        type="source"
        position={Position.Right}
        style={{ background: s.dot, border: `1px solid ${s.border}`, width: 8, height: 8, right: -5 }}
      />
    </>
  );
}

const nodeTypes = { workflowNode: WorkflowNode };

// ─── Initial graph state ──────────────────────────────────────────────────────
const INIT_NODES: { id: string; type: string; position: { x: number; y: number }; data: NodeData }[] = [
  { id: 'start',      type: 'workflowNode', position: { x: 30,  y: 260 }, data: { label: 'START',                    nodeType: 'start',  status: 'idle' } },
  { id: 'chatbot',    type: 'workflowNode', position: { x: 230, y: 260 }, data: { label: 'Chatbot / LLM',            subtitle: 'Qwen 3 32B · Groq',  nodeType: 'llm',    status: 'idle' } },
  { id: 'toolRouter', type: 'workflowNode', position: { x: 470, y: 260 }, data: { label: 'Tool Router',              subtitle: 'route_tools()',        nodeType: 'router', status: 'idle' } },
  { id: 'ragSearch',  type: 'workflowNode', position: { x: 720, y: 80  }, data: { label: 'rag_search',               subtitle: 'Hybrid RAG + Rerank', nodeType: 'tool',   status: 'idle' } },
  { id: 'getKpis',    type: 'workflowNode', position: { x: 720, y: 260 }, data: { label: 'get_kpis',                 subtitle: 'SQLite KPI DB',       nodeType: 'tool',   status: 'idle' } },
  { id: 'genReport',  type: 'workflowNode', position: { x: 720, y: 440 }, data: { label: 'generate_report_sections', subtitle: 'Multi-RAG Synthesis', nodeType: 'tool',   status: 'idle' } },
  { id: 'end',        type: 'workflowNode', position: { x: 470, y: 460 }, data: { label: 'END',                      nodeType: 'end',    status: 'idle' } },
];

const INIT_EDGES = [
  { id: 'start-chat',   source: 'start',      target: 'chatbot',    ...baseEdge() },
  { id: 'chat-tr',      source: 'chatbot',     target: 'toolRouter', label: 'has tool_calls?', labelStyle: { fill: '#4A5568', fontSize: 9 }, labelBgStyle: { fill: '#0D1117' }, ...baseEdge() },
  { id: 'tr-rag',       source: 'toolRouter',  target: 'ragSearch',  ...baseEdge() },
  { id: 'tr-kpi',       source: 'toolRouter',  target: 'getKpis',    ...baseEdge() },
  { id: 'tr-report',    source: 'toolRouter',  target: 'genReport',  ...baseEdge() },
  { id: 'rag-chat',     source: 'ragSearch',   target: 'chatbot',    type: 'smoothstep', ...baseEdge() },
  { id: 'kpi-chat',     source: 'getKpis',     target: 'chatbot',    type: 'smoothstep', ...baseEdge() },
  { id: 'report-chat',  source: 'genReport',   target: 'chatbot',    type: 'smoothstep', ...baseEdge() },
  { id: 'tr-end',       source: 'toolRouter',  target: 'end',        type: 'smoothstep', label: 'no tool_calls', labelStyle: { fill: '#4A5568', fontSize: 9 }, labelBgStyle: { fill: '#0D1117' }, ...baseEdge() },
];

// ─── Log colours ─────────────────────────────────────────────────────────────
const LOG_COLOR: Record<string, string> = {
  query_start:   '#68D391',
  node_enter:    '#90CDF4',
  tool_call:     '#F6AD55',
  tool_result:   '#76E4F7',
  node_exit:     '#B794F4',
  query_complete:'#68D391',
  error:         '#FC8181',
  ping:          '#2D3748',
};

// ─── Monitor Page ─────────────────────────────────────────────────────────────
let _logId = 0;

export default function MonitorPage() {
  const [nodes, setNodes, onNodesChange] = useNodesState(INIT_NODES);
  const [edges, setEdges, onEdgesChange] = useEdgesState(INIT_EDGES);

  const [connected, setConnected] = useState(false);
  const [currentQuery, setCurrentQuery]   = useState('');
  const [currentNode,  setCurrentNode]    = useState('');
  const [currentTool,  setCurrentTool]    = useState('');
  const [currentArgs,  setCurrentArgs]    = useState<Record<string, unknown>>({});
  const [outputPreview,setOutputPreview]  = useState('');
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll log feed
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // ── Helpers (functional-update form → no stale closures → [] deps valid) ──
  const addLog = useCallback((type: string, message: string) => {
    const now = new Date();
    const time = now.toLocaleTimeString('en-US', { hour12: false });
    setLogs(prev => [...prev.slice(-149), { id: ++_logId, time, type, message }]);
  }, []);

  const setNodeStatus = useCallback((id: string, status: NodeStatus) => {
    setNodes(ns => ns.map(n => n.id === id ? { ...n, data: { ...n.data, status } } : n));
  }, [setNodes]);

  const activateEdge = useCallback((id: string, active: boolean) => {
    setEdges(es => es.map(e => e.id === id ? applyEdge(e, active) : e));
  }, [setEdges]);

  const resetGraph = useCallback(() => {
    setNodes(ns => ns.map(n => ({ ...n, data: { ...n.data, status: 'idle' } })));
    setEdges(es => es.map(e => applyEdge(e, false)));
    setCurrentNode('');
    setCurrentTool('');
    setCurrentArgs({});
    setOutputPreview('');
  }, [setNodes, setEdges]);

  // ── Event handler ─────────────────────────────────────────────────────────
  const handleEvent = useCallback((ev: WorkflowEvent) => {
    if (ev.type === 'ping') return;
    addLog(ev.type, ev.log ?? `${ev.type}`);

    switch (ev.type) {

      case 'query_start':
        resetGraph();
        setCurrentQuery(ev.query ?? '');
        // Mark start as done; animate start→chatbot; put chatbot in running
        setNodeStatus('start', 'completed');
        activateEdge('start-chat', true);
        setNodeStatus('chatbot', 'running');
        setCurrentNode('chatbot');
        break;

      case 'node_enter':
        if (ev.node === 'chatbot') {
          setNodeStatus('chatbot', 'running');
          setCurrentNode('chatbot');
        }
        break;

      case 'tool_call': {
        const tm = ev.tool ? TOOL_MAP[ev.tool] : null;
        // chatbot finished reasoning → activate chatbot→toolRouter
        setNodeStatus('chatbot', 'completed');
        setNodeStatus('toolRouter', 'running');
        activateEdge('chat-tr', true);
        if (tm) {
          activateEdge(tm.fwdEdge, true);
          setNodeStatus(tm.nodeId, 'running');
        }
        setCurrentNode('tools');
        setCurrentTool(ev.tool ?? '');
        setCurrentArgs(ev.args ?? {});
        break;
      }

      case 'tool_result': {
        const tm = ev.tool ? TOOL_MAP[ev.tool] : null;
        if (tm) {
          setNodeStatus(tm.nodeId, 'completed');
          activateEdge(tm.backEdge, true);
        }
        setNodeStatus('toolRouter', 'completed');
        // chatbot receiving result
        setNodeStatus('chatbot', 'running');
        setCurrentNode('chatbot');
        setOutputPreview(ev.output_preview ?? '');
        break;
      }

      case 'node_exit':
        // Final answer — no more tool calls
        setNodeStatus('chatbot', 'completed');
        activateEdge('tr-end', true);
        setNodeStatus('toolRouter', 'completed');
        setNodeStatus('end', 'completed');
        setOutputPreview(ev.output_preview ?? '');
        setCurrentNode('end');
        break;

      case 'query_complete':
        setNodeStatus('chatbot', 'completed');
        setNodeStatus('end', 'completed');
        setCurrentNode('');
        setEdges(es => es.map(e => applyEdge(e, false)));
        break;

      case 'error':
        // Mark any running node as error
        setNodes(ns => ns.map(n =>
          (n.data as any).status === 'running'
            ? { ...n, data: { ...n.data, status: 'error' } }
            : n
        ));
        setEdges(es => es.map(e => applyEdge(e, false)));
        break;
    }
  }, [addLog, resetGraph, setNodeStatus, activateEdge, setCurrentQuery, setCurrentNode, setCurrentTool, setCurrentArgs, setOutputPreview, setNodes]);

  // ── SSE connection with auto-reconnect ────────────────────────────────────
  useEffect(() => {
    let es: EventSource | null = null;
    let retry: ReturnType<typeof setTimeout>;

    const connect = () => {
      es = new EventSource('http://localhost:8000/api/workflow-stream');
      es.onopen = () => {
        setConnected(true);
        addLog('ping', 'Connected to AURA workflow stream');
      };
      es.onmessage = (e) => {
        try { handleEvent(JSON.parse(e.data)); } catch { /* malformed */ }
      };
      es.onerror = () => {
        setConnected(false);
        es?.close();
        retry = setTimeout(connect, 3500);
      };
    };

    connect();
    return () => { es?.close(); clearTimeout(retry); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={{
      height: '100vh',
      background: '#0A0F1E',
      display: 'flex',
      flexDirection: 'column',
      fontFamily: "'Inter', 'Plus Jakarta Sans', sans-serif",
      overflow: 'hidden',
    }}>

      {/* ── Global styles ── */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; }
        body { margin: 0; }
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #2D3748; border-radius: 2px; }
        .react-flow__renderer { background: #0A0F1E !important; }
        .react-flow__controls { background: #161B27 !important; border: 1px solid #2D3748 !important; border-radius: 8px !important; }
        .react-flow__controls button { background: #161B27 !important; border-bottom: 1px solid #2D3748 !important; color: #718096 !important; }
        .react-flow__controls button:hover { background: #1E2535 !important; color: #A0AEC0 !important; }
        .react-flow__edge-textbg { fill: #0D1117 !important; }
        .react-flow__minimap { background: #161B27 !important; border: 1px solid #2D3748 !important; border-radius: 8px !important; }
      `}</style>

      {/* ── Header ── */}
      <header style={{
        height: 52,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 1.25rem',
        borderBottom: '1px solid #1A202C',
        background: 'rgba(10,15,30,0.98)',
        backdropFilter: 'blur(12px)',
        flexShrink: 0,
        zIndex: 10,
      }}>
        {/* Left */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.65rem', color: '#4A5568', letterSpacing: '0.12em' }}>AURA</span>
          <div style={{ width: 1, height: 14, background: '#2D3748' }} />
          <span style={{ fontSize: '0.9rem', fontWeight: 700, color: '#E2E8F0', letterSpacing: '-0.01em' }}>
            Agent Workflow Monitor
          </span>
          <span style={{
            fontSize: '0.6rem',
            padding: '2px 9px',
            borderRadius: 20,
            background: 'rgba(159,122,234,0.08)',
            border: '1px solid rgba(159,122,234,0.25)',
            color: '#B794F4',
            fontFamily: "'JetBrains Mono', monospace",
            letterSpacing: '0.06em',
          }}>LangGraph · ReAct</span>
        </div>

        {/* Right */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, fontSize: '0.73rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{
              width: 7, height: 7, borderRadius: '50%',
              background: connected ? '#38A169' : '#E53E3E',
              boxShadow: connected ? '0 0 7px #38A169' : 'none',
              transition: 'all 0.4s',
            }} />
            <span style={{ color: connected ? '#68D391' : '#FC8181', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.68rem' }}>
              {connected ? 'STREAM ACTIVE' : 'RECONNECTING…'}
            </span>
          </div>
          <span style={{ color: '#2D3748', fontSize: '0.65rem', fontFamily: "'JetBrains Mono', monospace" }}>
            :8000/api/workflow-stream
          </span>
          <a href="/" style={{
            fontSize: '0.72rem', color: '#4A5568', textDecoration: 'none',
            display: 'flex', alignItems: 'center', gap: 4,
            padding: '4px 10px', borderRadius: 6, border: '1px solid #2D3748',
            transition: 'all 0.2s',
          }}
            onMouseEnter={e => (e.currentTarget.style.color = '#A0AEC0')}
            onMouseLeave={e => (e.currentTarget.style.color = '#4A5568')}
          >
            ← Main App
          </a>
        </div>
      </header>

      {/* ── Body ── */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

        {/* ── React Flow Canvas ── */}
        <div style={{ flex: 1, position: 'relative' }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.18 }}
            proOptions={{ hideAttribution: true }}
            nodesDraggable
            nodesConnectable={false}
            elementsSelectable={false}
          >
            <Background
              variant={BackgroundVariant.Dots}
              color="#1A202C"
              gap={26}
              size={1.2}
            />
            <Controls showInteractive={false} />
          </ReactFlow>

          {/* Canvas legend overlay */}
          <div style={{
            position: 'absolute',
            bottom: 16,
            left: 16,
            display: 'flex',
            gap: 10,
            background: 'rgba(13,17,23,0.85)',
            backdropFilter: 'blur(8px)',
            border: '1px solid #1A202C',
            borderRadius: 8,
            padding: '6px 12px',
          }}>
            {(Object.entries(STATUS) as [NodeStatus, typeof STATUS[NodeStatus]][])
              .filter(([k]) => k !== 'skipped')
              .map(([st, cfg]) => (
                <div key={st} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                  <div style={{ width: 7, height: 7, borderRadius: '50%', background: cfg.dot }} />
                  <span style={{ fontSize: '0.6rem', color: '#4A5568', fontFamily: "'JetBrains Mono', monospace" }}>{st}</span>
                </div>
              ))}
          </div>
        </div>

        {/* ── Side Panel ── */}
        <aside style={{
          width: 310,
          flexShrink: 0,
          background: '#0D1117',
          borderLeft: '1px solid #1A202C',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}>

          {/* ── Execution Context ── */}
          <section style={{ padding: '14px 16px', borderBottom: '1px solid #1A202C', flexShrink: 0 }}>
            <SectionLabel>Execution Context</SectionLabel>
            <InfoRow label="Query">
              <span style={{ fontSize: '0.7rem', color: '#A0AEC0', lineHeight: 1.5, wordBreak: 'break-word', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                {currentQuery || '—'}
              </span>
            </InfoRow>
            <InfoRow label="Active Node">
              <Pill color={currentNode ? '#4299E1' : '#2D3748'}>{currentNode || 'idle'}</Pill>
            </InfoRow>
            {currentTool && (
              <InfoRow label="Tool">
                <Pill color="#ECC94B">{currentTool}</Pill>
              </InfoRow>
            )}
          </section>

          {/* ── Tool Args ── */}
          {Object.keys(currentArgs).length > 0 && (
            <section style={{ padding: '14px 16px', borderBottom: '1px solid #1A202C', flexShrink: 0 }}>
              <SectionLabel>Tool Arguments</SectionLabel>
              {Object.entries(currentArgs).map(([k, v]) => (
                <div key={k} style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: '0.6rem', color: '#4A5568', fontFamily: "'JetBrains Mono', monospace", marginBottom: 2 }}>{k}</div>
                  <div style={{
                    fontSize: '0.68rem', color: '#90CDF4',
                    fontFamily: "'JetBrains Mono', monospace",
                    background: '#161B27', padding: '4px 8px',
                    borderRadius: 5, wordBreak: 'break-all', lineHeight: 1.4,
                    border: '1px solid #1A202C',
                  }}>
                    {v === null || v === undefined ? 'null' : String(v).slice(0, 120)}
                  </div>
                </div>
              ))}
            </section>
          )}

          {/* ── Output Preview ── */}
          {outputPreview && (
            <section style={{ padding: '14px 16px', borderBottom: '1px solid #1A202C', flexShrink: 0 }}>
              <SectionLabel>Output Preview</SectionLabel>
              <div style={{
                fontSize: '0.68rem', color: '#718096',
                fontFamily: "'JetBrains Mono', monospace",
                background: '#161B27', padding: '8px',
                borderRadius: 6, lineHeight: 1.5,
                maxHeight: 90, overflowY: 'auto',
                border: '1px solid #1A202C',
              }}>
                {outputPreview.slice(0, 320)}{outputPreview.length > 320 ? '…' : ''}
              </div>
            </section>
          )}

          {/* ── Event Log ── */}
          <section style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{
              padding: '10px 16px 8px',
              borderBottom: '1px solid #1A202C',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexShrink: 0,
            }}>
              <SectionLabel style={{ marginBottom: 0 }}>Event Log</SectionLabel>
              <button
                onClick={() => setLogs([])}
                style={{
                  background: 'none', border: 'none',
                  color: '#4A5568', fontSize: '0.62rem',
                  cursor: 'pointer', fontFamily: "'JetBrains Mono', monospace",
                  padding: '2px 6px', borderRadius: 4,
                }}
                onMouseEnter={e => (e.currentTarget.style.color = '#718096')}
                onMouseLeave={e => (e.currentTarget.style.color = '#4A5568')}
              >clear</button>
            </div>

            <div style={{ flex: 1, overflowY: 'auto', padding: '8px 10px' }}>
              {logs.length === 0 ? (
                <div style={{ color: '#2D3748', fontSize: '0.68rem', fontFamily: "'JetBrains Mono', monospace", textAlign: 'center', paddingTop: 32 }}>
                  Waiting for events…
                </div>
              ) : (
                logs.map(log => (
                  <div key={log.id} style={{ display: 'flex', gap: 8, marginBottom: 4, lineHeight: 1.4 }}>
                    <span style={{
                      color: '#2D3748', fontSize: '0.62rem',
                      fontFamily: "'JetBrains Mono', monospace",
                      flexShrink: 0, minWidth: 56,
                    }}>{log.time}</span>
                    <span style={{
                      fontSize: '0.68rem',
                      fontFamily: "'JetBrains Mono', monospace",
                      color: LOG_COLOR[log.type] ?? '#718096',
                      wordBreak: 'break-word',
                    }}>{log.message}</span>
                  </div>
                ))
              )}
              <div ref={logsEndRef} />
            </div>
          </section>

          {/* ── Footer: node type legend ── */}
          <div style={{ padding: '8px 16px', borderTop: '1px solid #1A202C', flexShrink: 0 }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px 14px' }}>
              {Object.entries(NODE_TYPE).map(([key, cfg]) => (
                <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                  <span style={{ fontSize: 10, color: cfg.accent }}>{cfg.icon}</span>
                  <span style={{ fontSize: '0.58rem', color: '#4A5568', fontFamily: "'JetBrains Mono', monospace" }}>{key}</span>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

// ─── Micro-components ─────────────────────────────────────────────────────────

function SectionLabel({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div style={{
      fontSize: '0.6rem', color: '#4A5568',
      textTransform: 'uppercase', letterSpacing: '0.1em',
      fontWeight: 700, marginBottom: 10,
      ...style,
    }}>{children}</div>
  );
}

function InfoRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ fontSize: '0.6rem', color: '#4A5568', fontFamily: "'JetBrains Mono', monospace", marginBottom: 3 }}>{label}</div>
      {children}
    </div>
  );
}

function Pill({ children, color }: { children: React.ReactNode; color: string }) {
  return (
    <span style={{
      display: 'inline-block',
      fontSize: '0.68rem',
      fontFamily: "'JetBrains Mono', monospace",
      color,
      background: `${color}14`,
      border: `1px solid ${color}40`,
      borderRadius: 5,
      padding: '2px 8px',
    }}>{children}</span>
  );
}
