//
//  Modifers.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 17.01.21.
//

import SwiftUI

// MARK: Frozen Window modifier used for glass look with transparent background, not used on 19/03/21
struct FrozenWindowModifier: ViewModifier {
    func body(content: Content) -> some View {
        content
            // background gets frozen via the BlurView
            .background(BlurView(style: .systemThinMaterial))
            // create a border line which is in the same size as the rounded Rectangle
            .overlay(
                RoundedRectangle(cornerRadius: 18.0, style: .continuous)
                    .stroke(Color.white.opacity(0.6), lineWidth: 1)
            )
            // new shape to RoundedRectangle
            .clipShape(RoundedRectangle(cornerRadius: 18.0, style: .continuous))
            .shadow(color: Color.black.opacity(0.1), radius: 1, x: 0, y: 1)
            .shadow(color: Color.black.opacity(0.2), radius: 10, x: 0, y: 10)
    }
}

// MARK: Grey/white looking background with round-corners and shadow
struct offWhiteShadow: ViewModifier {
    
    var cornerRadius: CGFloat
    
    func body(content: Content) -> some View {
        content
            .background(Color("Offwhite"))
            .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .shadow(color: Color.black.opacity(0.2), radius: 20, x: 0, y: 5)
    }
}
