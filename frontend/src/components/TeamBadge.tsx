import { useEffect, useState } from 'react';

export default function TeamBadge({ src, name, size = 28 }: { src?: string | null; name: string; size?: number }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [src]);

  if (!src || failed) {
    return (
      <div style={{ width: size, height: size }}
        className="rounded-full bg-[#3a1245] border border-[#52275f] flex items-center justify-center text-[9px] font-bold text-[#d5c8dc] shrink-0">
        {name.slice(0, 3).toUpperCase()}
      </div>
    );
  }
  return (
    <img src={src} alt={name} width={size} height={size} onError={() => setFailed(true)}
      referrerPolicy="no-referrer" className="object-contain shrink-0" />
  );
}