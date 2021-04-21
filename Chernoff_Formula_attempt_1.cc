#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_raviart_thomas.h> // Raviart-Thomas fe is declared in this
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_bdm.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/base/timer.h>


#include <deal.II/fe/mapping_q_eulerian.h>


#include <fstream>
#include <iostream>

#include <math.h>
 
using namespace dealii;

//-------------------------------------
// Started on 19th April 2021: Use Nonlinear Chernoff Formula to solve parabolic problems. Test with different \beta functions (scheme as in Magnes, Nochetto, Verdi Energy error estimates for a linear scheme to approximate nonlinear parabolic problems).

//-------------------------------------
double h_x = 1.0/32, h_y = 1.0/32;
double mu = 1e0;
double t_step = 5e-3;
double T_max = 25*1e-2;
//int max_it = std::floor(5e-3/t_step), max_it_print = 50;
int max_it = T_max/t_step, max_it_print = 50;

class ChernoffFormula
{
    public :
    
    ChernoffFormula();
    void run();
    
    private :
    
    void make_grid();
    void setup_system(double t, int time_step_number);
    void assemble_system(double t);
    void solve();
    void compute_E(double t, int time_step_number);
    void output_results (int time_step_number);
    void compute_error (double t, int time_step_number);
    void iterate_and_solve();
    
    
    Triangulation<2> triangulation;
    DoFHandler<2> dof_handler_1;
    DoFHandler<2> dof_handler_0;
    FE_Q<2> fe_1; // 1 continuous linear elements
    FE_DGQ<2> fe_0; // 0 piecewise constant elements
    
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> system_rhs, solution_E, previous_solution_E, solution_T, previous_solution_T;
    double time, k;
    
    double global_L2_error_T_square, global_L2_error_E_square;
    
    FullMatrix<double> alpha_i_j_global, alpha_i_beta_j_global;
    
    AffineConstraints<double> constraints_1, constraints_0;
    
    ConvergenceTable convergence_table;

    
};

ChernoffFormula::ChernoffFormula() :  dof_handler_1(triangulation), fe_1(1), dof_handler_0(triangulation), fe_0(0), k(1*t_step), time(1*t_step)

{}

class rhs_f : public Function<2> // Taken to be a function of T or \beta(E)
{
    public :
    
    rhs_f() : Function<2>(1)
    {}
    
    virtual double value (const Point<2> &p, const unsigned int component = 0) const override;
};

double rhs_f::value (const Point<2> &p, const unsigned int component) const
{
    double t = this->get_time();
    double x = p[0], y = p[1];
    double pi = numbers::PI;
    
//    return std::exp(-t)*x*(1-x)*y*(1-y) - 2*std::exp(-t)*( y*(1-y) + x*(1-x) );
    
//    return std::exp(-t)*std::sin(pi*x)*y*(1-y) + std::exp(-t)*( -pi*pi*std::sin(pi*x)*y*(1-y)  - 2*std::sin(pi*x)  );
    
//    if (  ( std::exp(-2*pi*pi*t)*std::sin(pi*x)*std::sin(pi*y) < 1 ) && ( std::exp(-2*pi*pi*t)*std::sin(pi*x)*std::sin(pi*y) > 0 ) )
//    {
//        return -2*pi*pi*std::exp(-2*pi*pi*t)*std::sin(pi*x)*std::sin(pi*y);
//    }
//    else
//    {
//        return 0;
//    }
    
    return 0;
        
}


class Boundary_Values_T : public Function<2>
{
    public :
    
    Boundary_Values_T() : Function<2>(1)
    {}
    
    virtual double value (const Point<2> &p, const unsigned int component = 0) const override;
};

double Boundary_Values_T::value (const Point<2> &p, const unsigned int component) const
{
    double t = this->get_time();
    double x = p[0], y = p[1];
    double pi = numbers::PI;
    
    const double exp_value = std::exp(-t);

//    return std::exp(-2*pi*pi*t)*std::sin(pi*x)*std::sin(pi*y);
    
    double E;

    double phi = -x -y + 2*t + 0.1;

    if (phi >= 0)
    {
        E = 2*(std::exp(phi) - 1) + 1;
    }
    else
    {
        E = std::exp(phi) - 1;
    }

    if (E < 0)
    {
        return E;
    }
    else if (E > 1)
    {
        return E - 1;
    }
    else
    {
        return 0;
    }

//    return std::exp(-5*pi*pi*t)*std::sin(2*pi*x)*std::sin(pi*y);
    
}

class Flux_T : public Function<2>
{
    public :
    
    Flux_T() : Function<2>(1)
    {}
    
    virtual double value (const Point<2> &p, const unsigned int component = 0) const override;
};

double Flux_T::value (const Point<2> &p, const unsigned int component) const
{
    double t = this->get_time();
    double x = p[0], y = p[1];
    double pi = numbers::PI;
    
//    return std::exp(-t)*x*(1-x)*y*(1-y) - 2*std::exp(-t)*( y*(1-y) + x*(1-x) );
    
//    return std::exp(-t)*std::sin(pi*x)*y*(1-y) + std::exp(-t)*( -pi*pi*std::sin(pi*x)*y*(1-y)  - 2*std::sin(pi*x)  );
    
    return 0;
     
        
}

class initial_solution_E : public Function<2>
{
    public :
    
    initial_solution_E() : Function<2>(1)
    {}
    
    virtual double value (const Point<2> &p, const unsigned int component = 0) const override;
};

double initial_solution_E::value (const Point<2> &p, const unsigned int component) const
{
    double t = this->get_time();
    double x = p[0], y = p[1];
    double pi = numbers::PI;

//    return std::exp(-5*pi*pi*t)*std::sin(2*pi*x)*std::sin(pi*y);
    
    double phi = -x -y + 2*t + 0.1;

    if (phi >= 0)
    {
        return 2*(std::exp(phi) - 1) + 1;
    }
    else
    {
        return std::exp(phi) - 1;
    }

//    return -2*std::sin(pi*x)*std::sin(pi*y);
//
//    return -std::exp(-t)*x*(1-x)*y*(1-y);
//
//    return -std::exp(-t)*std::sin(pi*x)*y*(1-y);
//
//    return 3*std::exp(-8*pi*pi*t)*std::sin(2*pi*x)*std::sin(2*pi*y);
//
//    return std::exp(-2*pi*pi*t)*std::sin(pi*x)*std::sin(pi*y);
//
//    return -1;
    
}

class exact_T : public Function<2>
{
    public :
    
    exact_T() : Function<2>(1)
    {}
    
    virtual double value (const Point<2> &p, const unsigned int component = 0) const override;
};

double exact_T::value (const Point<2> &p, const unsigned int component) const
{
    double t = this->get_time();
    double x = p[0], y = p[1];
    double pi = numbers::PI;
    
    double E;

    double phi = -x -y + 2*t + 0.1;

    if (phi >= 0)
    {
        E = 2*(std::exp(phi) - 1) + 1;
    }
    else
    {
        E = std::exp(phi) - 1;
    }

    if (E < 0)
    {
        return E;
    }
    else if (E > 1)
    {
        return E - 1;
    }
    else
    {
        return 0;
    }

//    return std::exp(-5*pi*pi*t)*std::sin(2*pi*x)*std::sin(pi*y);
    
//    return -std::exp(-t)*x*(1-x)*y*(1-y);
    
//    return -std::exp(-t)*std::sin(pi*x)*y*(1-y);
    
//    if (  ( std::exp(-2*pi*pi*t)*std::sin(pi*x)*std::sin(pi*y) < 1 ) && ( std::exp(-2*pi*pi*t)*std::sin(pi*x)*std::sin(pi*y) > 0 ) )
//    {
//    return 0;
//    }
//    else if (  ( std::exp(-2*pi*pi*t)*std::sin(pi*x)*std::sin(pi*y) > 1 ) )
//    {
//        return std::exp(-2*pi*pi*t)*std::sin(pi*x)*std::sin(pi*y) - 1;
//    }
//    else
//    {
//        return std::exp(-2*pi*pi*t)*std::sin(pi*x)*std::sin(pi*y);
//    }
}


class exact_E : public Function<2>
{
    public :
    
    exact_E() : Function<2>(1)
    {}
    
    virtual double value (const Point<2> &p, const unsigned int component = 0) const override;
};

double exact_E::value (const Point<2> &p, const unsigned int component) const
{
    double t = this->get_time();
    double x = p[0], y = p[1];
    double pi = numbers::PI;
    
//    return std::exp(-5*pi*pi*t)*std::sin(2*pi*x)*std::sin(pi*y);
    
    double E;

    double phi = -x -y + 2*t + 0.1;

    if (phi >= 0)
    {
        return 2*(std::exp(phi) - 1) + 1;
    }
    else
    {
        return std::exp(phi) - 1;
    }
    
}


//---------------------- class, function definitions end -------------------------------
    
//---------------------- member functions definitions begin -------------------------------

void ChernoffFormula::make_grid()
{
//    GridGenerator::hyper_cube(triangulation, 0, 1);
//
//    triangulation.refine_global(2);
    
    std::vector<unsigned int> repititions(2);
    
    repititions[0] = 1.0/h_x;
    repititions[1] = 1.0/h_y;

    Point<2> p1 (0,0);
    Point<2> p2 (0.5,0.25);

    GridGenerator::subdivided_hyper_rectangle(triangulation, repititions,  p1, p2);
    
    // boundary id label - clockwise 1 2 3 4, 1 at top of square
    
//    for (const auto &cell : triangulation.cell_iterators())
//    {
//        for (unsigned int face_n = 0; face_n < GeometryInfo<2>::faces_per_cell; ++face_n)
//        {
//            const auto center = cell->face(face_n)->center();
//
//            if ( std::fabs(center(1) - (1)) < 1e-12 ) // if y == 1
//            {
//                cell->face(face_n)->set_boundary_id(1);
//            }
//
//        }
//
//    }
//
//    for (const auto &cell : triangulation.cell_iterators())
//    {
//        for (unsigned int face_n = 0; face_n < GeometryInfo<2>::faces_per_cell; ++face_n)
//        {
//            const auto center = cell->face(face_n)->center();
//
//            if ( std::fabs(center(0) - (1)) < 1e-12 ) // if x == 1
//            {
//                cell->face(face_n)->set_boundary_id(2);
//            }
//
//        }
//
//    }
//
//    for (const auto &cell : triangulation.cell_iterators())
//    {
//        for (unsigned int face_n = 0; face_n < GeometryInfo<2>::faces_per_cell; ++face_n)
//        {
//            const auto center = cell->face(face_n)->center();
//
//            if ( std::fabs(center(1) - (0)) < 1e-12 ) // if y == 0
//            {
//                cell->face(face_n)->set_boundary_id(3);
//            }
//
//        }
//
//    }
//
//    for (const auto &cell : triangulation.cell_iterators())
//    {
//        for (unsigned int face_n = 0; face_n < GeometryInfo<2>::faces_per_cell; ++face_n)
//        {
//            const auto center = cell->face(face_n)->center();
//
//            if ( std::fabs(center(0) - (0)) < 1e-12 ) // if x == 0
//            {
//                cell->face(face_n)->set_boundary_id(4);
//            }
//
//        }
//
//    }
    
    
    std::cout<<"Number of active_cells: "<<triangulation.n_active_cells()<<std::endl;
}

void ChernoffFormula::setup_system(double t, int time_step_number)
{
    dof_handler_0.distribute_dofs(fe_0);
    dof_handler_1.distribute_dofs(fe_1);
    
    constraints_0.clear(); constraints_0.close();
    constraints_1.clear(); constraints_1.close();
    
    DynamicSparsityPattern dynamic_sparsity_pattern (dof_handler_1.n_dofs(), dof_handler_1.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler_1, dynamic_sparsity_pattern, constraints_1, false);
    sparsity_pattern.copy_from (dynamic_sparsity_pattern);
    
    system_matrix.reinit (sparsity_pattern);
    system_matrix = 0;
    
    system_rhs.reinit (dof_handler_1.n_dofs());
    system_rhs = 0;
    
    solution_E.reinit (dof_handler_0.n_dofs());
    solution_E = 0;
    
    solution_T.reinit (dof_handler_1.n_dofs());
    solution_T = 0;
    
    if (time_step_number == 1)
    {
        
        previous_solution_E.reinit(dof_handler_0.n_dofs());
    
        QGauss<2> quad_formula(fe_0.degree+1);

        initial_solution_E initial_solution_object;
        initial_solution_object.set_time(t - k);
        
//        previous_solution_E = 9;

//        VectorTools::project(dof_handler,constraints, quad_formula, initial_solution_object, previous_solution_E);
        
        VectorTools::project(dof_handler_0, constraints_0, quad_formula, initial_solution_object, previous_solution_E);

//        constraints_0.distribute(previous_solution_E);
    }
        
}


void ChernoffFormula::assemble_system(double t)
{
    QGauss<2> quadrature_formula(2);
    QGauss<2-1> quadrature_formula_face(2);
    
    FEValues<2> fe_values_1 (fe_1, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FEFaceValues<2> fe_values_face_1 (fe_1, quadrature_formula_face, update_values | update_normal_vectors | update_quadrature_points | update_JxW_values);
    
    FEValues<2> fe_values_0 (fe_0, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FEFaceValues<2> fe_values_face_0 (fe_0, quadrature_formula_face, update_values | update_normal_vectors | update_quadrature_points | update_JxW_values);
    
    FullMatrix<double> cell_matrix (fe_1.dofs_per_cell, fe_1.dofs_per_cell);
    Vector<double> cell_rhs (fe_1.dofs_per_cell);
    
    DoFHandler<2>::active_cell_iterator cell_1 = dof_handler_1.begin_active(), endc_1 = dof_handler_1.end();
    DoFHandler<2>::active_cell_iterator cell_0 = dof_handler_0.begin_active();
    
    rhs_f rhs_f_object;
    
    rhs_f_object.set_time(t);
    
    std::vector<types::global_dof_index> local_dof_indices_1 (fe_1.dofs_per_cell);
    
    std::vector<double> previous_E_value (quadrature_formula.size()); // get values at the quadrature points using fe_values.get_function_values
    std::vector<double> beta_previous_E_value (quadrature_formula.size());
    std::vector<double> rhs_f_values (quadrature_formula.size());
    
    for (; cell_1 != endc_1; ++cell_0, ++cell_1)
    {
        fe_values_1.reinit(cell_1);
        fe_values_0.reinit(cell_0);
        
        cell_matrix = 0;
        cell_rhs = 0;
        
        (*cell_1).get_dof_indices (local_dof_indices_1);
        
        for (unsigned int q = 0; q < quadrature_formula.size(); ++q) // compute the cell matrix
        {
            for (unsigned int i = 0; i < fe_1.dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < fe_1.dofs_per_cell; ++j)
                {
                    cell_matrix(i,j) += ( fe_values_1.shape_value(i,q) * fe_values_1.shape_value(j,q) )*fe_values_1.JxW(q) + (k/mu)*( fe_values_1.shape_grad(i,q) * fe_values_1.shape_grad(j,q) )*fe_values_1.JxW(q);
                }
            }
        }
        
        rhs_f_object.value_list (fe_values_1.get_quadrature_points(), rhs_f_values); // get source f values
        fe_values_0.get_function_values (previous_solution_E, previous_E_value); // get previous E values
  
        for (unsigned int i = 0; i < previous_E_value.size(); ++i)
        {
            if (previous_E_value[i] < 0)
            {
                beta_previous_E_value[i] = previous_E_value[i];
            }
            else if (previous_E_value[i] > 1)
            {
                beta_previous_E_value[i] = previous_E_value[i] - 1;
            }
            else
            {
                beta_previous_E_value[i] = 0;
            }
        }
        
//        for (unsigned int i = 0; i < previous_E_value.size(); ++i)
//        {
//            beta_previous_E_value[i] = (previous_E_value[i]); // \beta(E) = E;
//        }
        
        
        for (unsigned int q = 0; q < quadrature_formula.size(); ++q) // compute the cell rhs
        {
            for (unsigned int i = 0; i < fe_1.dofs_per_cell; ++i)
            {
                cell_rhs(i) += beta_previous_E_value[q] * fe_values_1.shape_value(i,q) * fe_values_1.JxW(q);
                cell_rhs(i) += (k/mu) * rhs_f_values[q] * fe_values_1.shape_value(i,q) * fe_values_1.JxW(q);
            }
        }
        
        Boundary_Values_T boundary_values_object; // Dirichlet boundary values for temperature T
        boundary_values_object.set_time(t);
        
        constraints_1.clear();
        
        VectorTools::interpolate_boundary_values (dof_handler_1, 0, boundary_values_object, constraints_1);
//        VectorTools::interpolate_boundary_values (dof_handler_1, 2, boundary_values_object, constraints_1);
//        VectorTools::interpolate_boundary_values (dof_handler_1, 3, boundary_values_object, constraints_1);
//        VectorTools::interpolate_boundary_values (dof_handler_1, 4, boundary_values_object, constraints_1);
        
        constraints_1.close();
        
        constraints_1.distribute_local_to_global (cell_matrix, cell_rhs, local_dof_indices_1, system_matrix, system_rhs);

    }
    
}

void ChernoffFormula::solve() // solve for temperature T
{
    SparseDirectUMFPACK A_direct_t;
    A_direct_t.initialize(system_matrix);

    A_direct_t.vmult(solution_T, system_rhs);
    
    constraints_1.distribute(solution_T);
}

void ChernoffFormula::compute_E(double t, int time_step_number) // compute E using T
{
    QGauss<2> quadrature_formula(1); // compute T at the cell center
    
    FEValues<2> fe_values_1 (fe_1, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    
    FEValues<2> fe_values_0 (fe_0, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);

    
    DoFHandler<2>::active_cell_iterator cell_1 = dof_handler_1.begin_active(), endc_1 = dof_handler_1.end();
    DoFHandler<2>::active_cell_iterator cell_0 = dof_handler_0.begin_active();
    
    std::vector<types::global_dof_index> local_dof_indices_0 (fe_0.dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_1 (fe_1.dofs_per_cell);
    
    std::vector<double> temperature_value (quadrature_formula.size()); // store T at the cell center
    
    for (; cell_1 != endc_1; ++cell_0, ++cell_1)
    {
        fe_values_0.reinit(cell_0);
        fe_values_1.reinit(cell_1);
        
        (*cell_0).get_dof_indices (local_dof_indices_0);
        
        fe_values_1.get_function_values (solution_T, temperature_value); // get T at the cell center using 1 point quadrature_formula defined
        
        double beta_E_previous_value, previous_E_value = (previous_solution_E(local_dof_indices_0[0]));
        
        {
            if (previous_E_value < 0)
            {
                beta_E_previous_value = previous_E_value;
            }
            else if (previous_E_value > 1)
            {
                beta_E_previous_value = previous_E_value - 1;
            }
            else
            {
                beta_E_previous_value = 0;
            }
        }
        
        
//        beta_E_previous_value = ( (previous_solution_E(local_dof_indices_0[0])) ); // \beta(E) = E

        solution_E(local_dof_indices_0[0]) = previous_solution_E(local_dof_indices_0[0]) + mu*(temperature_value[0] - beta_E_previous_value);
        
    }
    
}
    
  

void ChernoffFormula::output_results(int time_step_number)
{
    std::vector<std::string> solution_names_t;

    solution_names_t.push_back("temperature");
    
    DataOut<2> data_out_t;
        
    data_out_t.attach_dof_handler (dof_handler_1);
        
    data_out_t.add_data_vector (dof_handler_1, solution_T, solution_names_t);
        
    data_out_t.build_patches(fe_1.degree);
        
    if (time_step_number <= max_it_print)
    {
        
        std::ofstream output_1("Chernoff_Formula_attempt_1-" + std::to_string(time_step_number) + ".vtu");
        data_out_t.write_vtu(output_1);

    }
    
    static std::vector<std::pair<double, std::string>> times_and_names_t;
        
    const std::string filename_t = "Chernoff_Formula_attempt_1-" +Utilities::int_to_string (time_step_number, 3) +".vtu";
     
    times_and_names_t.emplace_back (time, filename_t);
     
    std::ofstream output_1_t(filename_t);
     
    data_out_t.write_vtu(output_1_t);
         
    std::ofstream pvd_output_t ("Chernoff_Formula_attempt_1_T.pvd");
     
    DataOutBase::write_pvd_record (pvd_output_t, times_and_names_t);
    
    // E ----
    
    std::vector<std::string> solution_names_E;

    solution_names_E.push_back("enthalpy");
    
    DataOut<2> data_out_E;
        
    data_out_E.attach_dof_handler (dof_handler_0);
        
    data_out_E.add_data_vector (dof_handler_0, solution_E, solution_names_E);
        
    data_out_E.build_patches(fe_0.degree);
        
    if (time_step_number <= max_it_print)
    {
        
        std::ofstream output_2("Chernoff_Formula_attempt_1_E-" + std::to_string(time_step_number) + ".vtu");
        data_out_E.write_vtu(output_2);

    }
    
    static std::vector<std::pair<double, std::string>> times_and_names_E;
        
    const std::string filename_E = "Chernoff_Formula_attempt_1_E-" +Utilities::int_to_string (time_step_number, 3) +".vtu";
     
    times_and_names_E.emplace_back (time, filename_E);
     
    std::ofstream output_2_E(filename_E);
     
    data_out_E.write_vtu(output_2_E);
         
    std::ofstream pvd_output_E ("Chernoff_Formula_attempt_1_E.pvd");
     
    DataOutBase::write_pvd_record (pvd_output_E, times_and_names_E);
     
}

void ChernoffFormula::compute_error (double t, int time_step_number)
{
    if (time_step_number == 1)
    {
        global_L2_error_T_square = 0;
        global_L2_error_E_square = 0;
    }
    
    Vector<float> difference_per_cell(triangulation.n_active_cells());
    Vector<float> difference_per_cell_E(triangulation.n_active_cells());
    
    exact_T exact_T_object;
    exact_E exact_E_object;
    
    exact_T_object.set_time(t);
    exact_E_object.set_time(t);
    
    VectorTools::integrate_difference (dof_handler_1, solution_T, exact_T_object, difference_per_cell, QGauss<2>(fe_1.degree + 2), VectorTools::L2_norm);
    VectorTools::integrate_difference (dof_handler_0, solution_E, exact_E_object, difference_per_cell_E, QGauss<2>(fe_0.degree+2), VectorTools::L2_norm);
    
    const double L2_error_T = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);
    const double L2_error_E = VectorTools::compute_global_error(triangulation, difference_per_cell_E, VectorTools::L2_norm);
    
    global_L2_error_T_square += k * L2_error_T * L2_error_T;
    global_L2_error_E_square += k * L2_error_E * L2_error_E;
    
    convergence_table.add_value("time_step_number",time_step_number);
    
    convergence_table.add_value("1/h_x",1./h_x);
    
    convergence_table.add_value("1/h_y",1./h_y);
    
    convergence_table.add_value("time",time);
    
    convergence_table.add_value("#dofs",dof_handler_1.n_dofs());
    
    convergence_table.add_value("#cells",triangulation.n_active_cells());
    
    convergence_table.add_value("L2_T_error",L2_error_T);
    
    convergence_table.add_value("L2_E_error",L2_error_E);
    
    if (time_step_number == max_it)
    {
        convergence_table.set_precision("time",3); // 3 digit precision
  
        convergence_table.set_scientific("time",true);
    
        convergence_table.set_precision("L2_T_error",3); // 3 digit precision
 
        convergence_table.set_scientific("L2_T_error",true);
        
        convergence_table.set_precision("L2_E_error",3); // 3 digit precision
 
        convergence_table.set_scientific("L2_E_error",true);
        
        convergence_table.set_tex_caption("#cells","\\# cells");
  
        convergence_table.set_tex_caption("#dofs","\\# dofs");
 
        convergence_table.set_tex_caption("L2_T_error","L^2 T error");
        
        convergence_table.set_tex_caption("L2_E_error","L^2 E error");
    
        convergence_table.write_text(std::cout); // display the table
        
        std::cout<<"---- ---- ---- ----"<<std::endl;
        
        std::cout<<std::setprecision(9)<<"L2_L2_error_T is: "<<std::sqrt(global_L2_error_T_square)<<std::endl;
        std::cout<<std::setprecision(9)<<"L2_L2_error_E is: "<<std::sqrt(global_L2_error_E_square)<<std::endl;
    }
}

void ChernoffFormula::iterate_and_solve()
{
    int time_step_number = 1;

    while (time_step_number <= max_it)
    {
        setup_system(time, time_step_number);
        assemble_system(time);
        solve();
        compute_E(time, time_step_number);
        
        output_results(time_step_number);
        
        compute_error(time, time_step_number);
        
        previous_solution_E = solution_E;
        
        time_step_number += 1;
        
        time += k;
        
        std::cout<<"time_step_number: "<<time_step_number<<std::endl;
    }
}

void ChernoffFormula::run()
{
    make_grid();
    iterate_and_solve();
}

int main()
{
    Timer timer;
    
    ChernoffFormula ChernoffFormula_attempt_1;
    
    ChernoffFormula_attempt_1.run();
    
    timer.stop();
    
    std::cout<< "Time taken: (" << timer.cpu_time() << "s)" << std::endl;
    
    std::cout<<"* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- *"<<std::endl;
}
