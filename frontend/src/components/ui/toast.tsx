"use client"

import * as React from "react"
import { createContext, useContext, useState } from "react"
import { toast as sonnerToast, Toaster as SonnerToaster } from "sonner"

type ToastProps = React.ComponentProps<typeof sonnerToast>

type ToastActionElement = React.ReactElement<unknown>

type ToastVariant = "default" | "destructive"

export type Toast = {
  id?: string
  title?: React.ReactNode
  description?: React.ReactNode
  action?: ToastActionElement
  variant?: ToastVariant
}

const ToastContext = createContext<{
  toast: (props: Toast) => void
  dismiss: (toastId?: string) => void
}>({
  toast: () => {},
  dismiss: () => {},
})

export const useToast = () => {
  const context = useContext(ToastContext)
  if (!context) {
    throw new Error("useToast must be used within a ToastProvider")
  }
  return context
}

export function ToastProvider({
  children,
}: {
  children: React.ReactNode
}) {
  const [, forceUpdate] = useState({})

  const toast = React.useCallback(
    ({ title, description, variant, action, ...props }: Toast) => {
      sonnerToast(title as string, {
        description,
        action,
        className: variant === "destructive" ? "destructive" : "",
        ...props,
      })
      // Force a re-render to ensure the toast is displayed
      forceUpdate({})
    },
    []
  )

  const dismiss = React.useCallback((toastId?: string) => {
    sonnerToast.dismiss(toastId)
  }, [])

  return (
    <ToastContext.Provider value={{ toast, dismiss }}>
      {children}
    </ToastContext.Provider>
  )
}

export { SonnerToaster as Toaster } 