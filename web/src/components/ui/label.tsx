import * as React from 'react'

import { cn } from '@/lib/utils'

const Label = React.forwardRef<HTMLLabelElement, React.LabelHTMLAttributes<HTMLLabelElement>>(
  ({ className, ...props }, ref) => (
    <label
      ref={ref}
      className={cn('font-mono text-[10px] uppercase tracking-[0.18em] text-muted', className)}
      {...props}
    />
  ),
)
Label.displayName = 'Label'

export { Label }
