import { OrbitControls } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import { useEffect, useMemo } from 'react';
import * as THREE from 'three';
import type { TransactionNetwork, TransactionNetworkNode } from './graph';
import { relatedNodeIds } from './graph';

const riskColors: Record<string, string> = {
  LOW: '#347052',
  GUARDED: '#73858d',
  ELEVATED: '#986018',
  HIGH: '#c25231',
  CRITICAL: '#b33b35',
};

const relationColors = {
  entity: '#367b7d',
  merchant: '#65787f',
  device: '#8a7045',
} as const;

function EdgeSegments({
  network,
  selectedId,
}: {
  network: TransactionNetwork;
  selectedId: string | null;
}) {
  const nodeMap = useMemo(() => new Map(network.nodes.map((node) => [node.id, node])), [network]);
  const [active, context] = useMemo(() => {
    const activePoints: number[] = [];
    const contextPoints: number[] = [];
    const selectedNodeId = selectedId ? `transaction:${selectedId}` : null;
    network.edges.forEach((edge) => {
      const source = nodeMap.get(edge.source);
      const target = nodeMap.get(edge.target);
      if (!source || !target) return;
      const targetArray =
        !selectedNodeId || edge.source === selectedNodeId ? activePoints : contextPoints;
      targetArray.push(...source.position, ...target.position);
    });
    return [activePoints, contextPoints];
  }, [network, nodeMap, selectedId]);

  return (
    <>
      {context.length > 0 && (
        <lineSegments>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[new Float32Array(context), 3]} />
          </bufferGeometry>
          <lineBasicMaterial color="#9caaa8" transparent opacity={0.12} />
        </lineSegments>
      )}
      {active.length > 0 && (
        <lineSegments>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[new Float32Array(active), 3]} />
          </bufferGeometry>
          <lineBasicMaterial color="#558d8e" transparent opacity={selectedId ? 0.72 : 0.36} />
        </lineSegments>
      )}
    </>
  );
}

function NetworkNode({
  node,
  active,
  selected,
  onSelect,
}: {
  node: TransactionNetworkNode;
  active: boolean;
  selected: boolean;
  onSelect: (transactionId: string) => void;
}) {
  const transaction = node.transaction;
  const risk = transaction?.prediction.risk;
  const color = risk
    ? riskColors[risk.level]
    : relationColors[node.kind as keyof typeof relationColors];
  const nodeColor = color ?? '#65787f';
  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: nodeColor,
        emissive: selected ? nodeColor : '#000000',
        emissiveIntensity: selected ? 0.42 : 0,
        opacity: active ? 0.96 : 0.2,
        transparent: true,
        roughness: 0.62,
        metalness: 0.08,
      }),
    [active, nodeColor, selected],
  );
  useEffect(() => () => material.dispose(), [material]);
  const size = transaction ? 0.22 + transaction.prediction.risk.score * 0.34 : 0.27;
  const click = transaction ? () => onSelect(transaction.prediction.transaction_id) : undefined;

  return (
    <group position={node.position}>
      <mesh material={material} {...(click ? { onClick: click } : {})} scale={selected ? 1.18 : 1}>
        {node.kind === 'transaction' ? (
          <sphereGeometry args={[size, 18, 14]} />
        ) : node.kind === 'entity' ? (
          <octahedronGeometry args={[size * 1.12, 0]} />
        ) : node.kind === 'merchant' ? (
          <boxGeometry args={[size * 1.5, size * 1.5, size * 1.5]} />
        ) : (
          <cylinderGeometry args={[size * 0.8, size * 0.8, size * 1.5, 12]} />
        )}
      </mesh>
      {selected && (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[size * 1.55, 0.045, 10, 28]} />
          <meshBasicMaterial color="#fffefa" transparent opacity={0.9} />
        </mesh>
      )}
    </group>
  );
}

function Scene({
  network,
  selectedId,
  onSelect,
}: {
  network: TransactionNetwork;
  selectedId: string | null;
  onSelect: (transactionId: string) => void;
}) {
  const related = useMemo(() => relatedNodeIds(network, selectedId), [network, selectedId]);
  return (
    <>
      <color attach="background" args={['#17242a']} />
      <ambientLight intensity={1.6} />
      <directionalLight position={[5, 8, 4]} intensity={2.2} color="#e6f5f1" />
      <directionalLight position={[-5, 2, -4]} intensity={0.7} color="#d7aa5f" />
      <gridHelper args={[15, 15, '#385057', '#24373e']} position={[0, -1.15, 0]} />
      <EdgeSegments network={network} selectedId={selectedId} />
      {network.nodes.map((node) => (
        <NetworkNode
          key={node.id}
          node={node}
          active={!selectedId || related.has(node.id)}
          selected={
            node.kind === 'transaction' &&
            node.transaction?.prediction.transaction_id === selectedId
          }
          onSelect={onSelect}
        />
      ))}
      <OrbitControls
        makeDefault
        enablePan={false}
        enableDamping
        dampingFactor={0.08}
        minDistance={5.5}
        maxDistance={17}
        minPolarAngle={0.4}
        maxPolarAngle={Math.PI * 0.48}
      />
    </>
  );
}

export default function TransactionNetworkScene({
  network,
  selectedId,
  onSelect,
  resetKey,
  active,
}: {
  network: TransactionNetwork;
  selectedId: string | null;
  onSelect: (transactionId: string) => void;
  resetKey: number;
  active: boolean;
}) {
  return (
    <Canvas
      key={resetKey}
      aria-hidden="true"
      camera={{ position: [7.5, 7.2, 9.5], fov: 46, near: 0.1, far: 70 }}
      dpr={[1, 1.5]}
      frameloop={active ? 'demand' : 'never'}
      gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
    >
      <Scene network={network} selectedId={selectedId} onSelect={onSelect} />
    </Canvas>
  );
}
