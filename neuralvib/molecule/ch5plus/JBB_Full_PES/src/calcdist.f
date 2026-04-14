      SUBROUTINE calcdist(x1,x2,dist)
!.... a subroutine to calculate the bond length
!.... x1, x2 are Cartisian coordinate
!.... 
      implicit none

      double precision, dimension(3) :: x1, x2
      double precision :: dist

      dist = dsqrt((x1(1) - x2(1))**2 + (x1(2) - x2(2))**2
     $     + (x1(3) - x2(3))**2)

      END SUBROUTINE calcdist
